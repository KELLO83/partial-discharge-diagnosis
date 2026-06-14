import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Cpu,
  Database,
  Download,
  FileSearch,
  FileCheck2,
  FileImage,
  FileJson,
  FileText,
  Gauge,
  Image as ImageIcon,
  Loader2,
  Printer,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import { Fragment, useEffect, useId, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  apiAssetUrl,
  fetchDiagnosisHistory,
  fetchDiagnosisDetail,
  fetchDiagnosisReport,
  fetchDiagnosisTrace,
  fetchHealth,
  fetchModelRuntimeStatus,
  fetchRagDocuments,
  fetchRagQueryLogs,
  fetchRagStatus,
  fetchReviewQueue,
  reindexRagDocuments,
  searchRagDocuments,
  submitDiagnosis,
} from "./api";
import { fileDisplayText } from "./fileDisplay";
import { buildInputPresence } from "./formState";
import { DEFAULT_METADATA } from "./metadataDefaults";
import { selectInputRoute } from "./route";
import type {
  DiagnosisDetailResponse,
  DiagnosisListItem,
  DiagnosisResponse,
  DiagnosisStatus,
  EvidenceFactor,
  MetadataForm,
  ModelRuntimeStatus,
  RagDocument,
  RagDocumentListItem,
  RagQueryLogItem,
  RagReindexResponse,
  RagStatusResponse,
  SimilarCase,
  FusionSummaryPayload,
  StandardModelEvidence,
  TraceResponse,
} from "./types";
import "./styles.css";

type RequestStatus = "idle" | "submitting" | "failed";
type BackendStatus = "checking" | "online" | "offline";
type RagPanelStatus = "loading" | "idle" | "searching" | "reindexing" | "failed";

const RAG_TOP_K_MIN = 1;
const RAG_TOP_K_MAX = 20;
const RAG_DATASET_LIMIT_MIN = 1;
const RAG_DATASET_LIMIT_MAX = 50_000;

type SignalAnomalyRegion = {
  readonly frame: number;
  readonly start_index: number;
  readonly end_index: number;
  readonly count: number;
  readonly peak_abs: number;
};

type SignalSummary = {
  readonly frame_count: number;
  readonly channel_count: number;
  readonly sample_count: number;
  readonly mean: number;
  readonly rms: number;
  readonly peak_abs: number;
  readonly p99_abs: number;
  readonly anomaly_threshold: number;
  readonly anomaly_count: number;
  readonly anomaly_rate: number;
  readonly anomaly_regions: readonly SignalAnomalyRegion[];
};

type InputArtifactEvidence = {
  readonly prpdImageUrl: string | null;
  readonly timeseriesCsvUrl: string | null;
  readonly signalSummary: SignalSummary | null;
};

type ModelSignal = {
  readonly source: string;
  readonly label: string;
  readonly confidence: number | null;
  readonly detail: string;
};

const RAG_ADMIN_LIST_LIMIT = 5;

const routeLabel = {
  hybrid: "종합 진단",
  timeseries_only: "시계열 진단",
  vlm_only: "비전 진단",
  insufficient_input: "입력 대기",
};

const statusLabel: Record<DiagnosisStatus, string> = {
  completed: "완료",
  needs_review: "검토",
  rejected: "반려",
};

export function App() {
  const [image, setImage] = useState<File | null>(null);
  const [csv, setCsv] = useState<File | null>(null);
  const [metadataJsonStatus, setMetadataJsonStatus] = useState("메타데이터 JSON 업로드 대기");
  const [metadata, setMetadata] = useState<MetadataForm>(DEFAULT_METADATA);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [detail, setDetail] = useState<DiagnosisDetailResponse | null>(null);
  const [history, setHistory] = useState<readonly DiagnosisListItem[]>([]);
  const [reviewQueue, setReviewQueue] = useState<readonly DiagnosisListItem[]>([]);
  const [selectedHistory, setSelectedHistory] = useState<DiagnosisListItem | null>(null);
  const [expandedTraceKey, setExpandedTraceKey] = useState<string | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [requestStatus, setRequestStatus] = useState<RequestStatus>("idle");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [modelRuntime, setModelRuntime] = useState<ModelRuntimeStatus | null>(null);
  const route = useMemo(
    () => selectInputRoute(buildInputPresence({ hasImage: image !== null, hasTimeseries: csv !== null, metadata })),
    [image, csv, metadata],
  );
  const activeRoute = result?.route ?? selectedHistory?.route ?? route;
  const verdictStatus = result?.status ?? selectedHistory?.status;
  const currentSimilarCases = useMemo(() => similarCasesFromTrace(trace), [trace]);

  useEffect(() => {
    let active = true;
    void loadDashboard()
      .then((dashboard) => {
        if (active) {
          setBackendStatus(dashboard.backendStatus);
          setHistory(dashboard.history);
          setReviewQueue(dashboard.reviewQueue);
          setModelRuntime(dashboard.modelRuntime);
        }
      })
      .catch(() => {
        if (active) {
          setBackendStatus("offline");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (image === null) {
      setImagePreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(image);
    setImagePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setRequestStatus("submitting");
    setTrace(null);
    try {
      const response = await submitDiagnosis({ image, csv, metadata });
      setResult(response);
      setSelectedHistory(null);
      const [nextTrace, nextDetail] = await Promise.all([
        fetchDiagnosisTrace(response.diagnosis_id),
        fetchDiagnosisDetail(response.diagnosis_id),
      ]);
      setTrace(nextTrace);
      setDetail(nextDetail);
      await refreshLists().catch(() => undefined);
      setRequestStatus("idle");
    } catch {
      setRequestStatus("failed");
    }
  }

  async function handleMetadataJson(file: File | null): Promise<void> {
    if (file === null) {
      setMetadataJsonStatus("메타데이터 JSON 업로드 대기");
      return;
    }
    try {
      const json = JSON.parse(await file.text()) as unknown;
      setMetadata((current) => metadataFromJson(json, current));
      setMetadataJsonStatus(`${file.name} 적용 완료`);
    } catch {
      setMetadataJsonStatus(`${file.name} 적용 실패`);
    }
  }

  async function refreshLists(): Promise<void> {
    const [historyResponse, queueResponse] = await Promise.all([
      fetchDiagnosisHistory(),
      fetchReviewQueue(),
    ]);
    setHistory(historyResponse.items);
    setReviewQueue(queueResponse.items);
  }

  async function openDiagnosis(item: DiagnosisListItem): Promise<void> {
    setSelectedHistory(item);
    setResult(null);
    setExpandedTraceKey(null);
    const nextDetail = await fetchDiagnosisDetail(item.diagnosis_id);
    setTrace(nextDetail.trace);
    setDetail(nextDetail);
  }

  async function downloadReport(): Promise<void> {
    const diagnosisId = detail?.diagnosis.diagnosis_id ?? result?.diagnosis_id;
    if (diagnosisId === undefined) {
      return;
    }
    const report = await fetchDiagnosisReport(diagnosisId);
    downloadJson(`${diagnosisId}-diagnosis-report.json`, report);
  }

  return (
    <main className="shell">
      <aside className="sidebar" aria-label="주 메뉴">
        <div className="brandBlock">
          <div className="brandMark">PD</div>
          <div>
            <p className="eyebrow">산업 설비</p>
            <h1>부분방전 진단</h1>
          </div>
        </div>
        <div className="plantBadge">
          <span>공정 현장</span>
          <strong>설비 진단 콘솔</strong>
        </div>
        <nav className="navList">
          <a href="#overview"><Gauge size={18} />현황</a>
          <a href="#intake"><Database size={18} />진단 접수</a>
          <a href="#verdict"><ShieldCheck size={18} />판정</a>
          <a href="#evidence"><BarChart3 size={18} />근거</a>
          <a href="#case-search"><ImageIcon size={18} />유사 사례</a>
          <a href="#detail"><FileCheck2 size={18} />상세</a>
          <a href="#report"><FileSearch size={18} />리포트</a>
          <a href="#rag-admin"><BookOpen size={18} />RAG 관리</a>
          <a href="#history"><Database size={18} />이력</a>
          <a href="#queue"><AlertTriangle size={18} />검토 큐</a>
          <a href="#trace"><Cpu size={18} />추적</a>
          <a href="#model-runtime"><Cpu size={18} />모델 상태</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">설비 관측 콘솔</p>
            <h2>부분방전 종합 진단</h2>
          </div>
          <button className="iconButton" type="button" onClick={() => window.location.reload()}>
            <RefreshCw size={18} />
            새로고침
          </button>
        </header>

        <section id="overview" className="kpiGrid">
          <MetricCard icon={<Database size={20} />} label="서버" value={backendStatusLabel(backendStatus)} tone={backendTone(backendStatus)} />
          <MetricCard icon={<Gauge size={20} />} label="진단 경로" value={routeLabel[activeRoute]} tone="blue" />
          <MetricCard icon={<BarChart3 size={20} />} label="진단 이력" value={history.length.toString()} tone="amber" />
          <MetricCard icon={<Cpu size={20} />} label="현재 유사 사례" value={currentSimilarCases.length.toString()} tone="violet" />
        </section>

        <section className="twoColumn">
          <Panel id="intake" title="진단 접수" action={routeLabel[route]}>
            <form className="intakeForm" onSubmit={(event) => void handleSubmit(event)}>
              <div className="uploadGrid">
                <UploadField
                  accept="image/png"
                  description={fileDisplayText(image, "PRPD 이미지")}
                  icon={<FileImage size={20} />}
                  label="PRPD 이미지"
                  onChange={setImage}
                />
                <UploadField
                  accept=".csv,text/csv"
                  description={fileDisplayText(csv, "시계열 CSV")}
                  icon={<FileText size={20} />}
                  label="시계열 CSV"
                  onChange={setCsv}
                />
                <UploadField
                  accept=".json,application/json"
                  description={metadataJsonStatus}
                  icon={<FileJson size={20} />}
                  label="메타데이터 JSON"
                  onChange={(file) => void handleMetadataJson(file)}
                />
              </div>

              <InputInspector csv={csv} image={image} imagePreviewUrl={imagePreviewUrl} />

              <div className="formSection">
                <div className="sectionTitle">설비 정보</div>
                <div className="fieldGrid">
                  <Field label="설비명" value={metadata.equipmentName} onChange={(value) => setMetadata({ ...metadata, equipmentName: value })} />
                  <Field label="설비 유형" value={metadata.equipmentType ?? ""} onChange={(value) => setMetadata({ ...metadata, equipmentType: value })} />
                  <Field label="정격 전압" value={metadata.ratedVoltage} onChange={(value) => setMetadata({ ...metadata, ratedVoltage: value })} />
                  <Field label="정격 전류" value={metadata.ratedCurrent} onChange={(value) => setMetadata({ ...metadata, ratedCurrent: value })} />
                  <Field label="센서" value={metadata.sensorType} onChange={(value) => setMetadata({ ...metadata, sensorType: value })} />
                  <Field label="측정 위치" value={metadata.measurementLocation ?? ""} onChange={(value) => setMetadata({ ...metadata, measurementLocation: value })} />
                  <Field label="운전 상태" value={metadata.operatingCondition ?? ""} onChange={(value) => setMetadata({ ...metadata, operatingCondition: value })} />
                  <Field label="온도" value={metadata.temperature} onChange={(value) => setMetadata({ ...metadata, temperature: value })} />
                  <Field label="습도" value={metadata.humidity} onChange={(value) => setMetadata({ ...metadata, humidity: value })} />
                  <Field label="절연 유형" value={metadata.insulatorType ?? ""} onChange={(value) => setMetadata({ ...metadata, insulatorType: value })} />
                  <Field label="이격 거리" value={metadata.clearanceDistance ?? ""} onChange={(value) => setMetadata({ ...metadata, clearanceDistance: value })} />
                </div>
              </div>

              <div className="formFooter">
                <div>
                  <span>선택된 진단 경로</span>
                  <strong>{routeLabel[route]}</strong>
                </div>
                <button className="primaryButton" type="submit" disabled={requestStatus === "submitting"}>
                  {requestStatus === "submitting" ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
                  {requestStatus === "submitting" ? "진단 중" : "진단 실행"}
                </button>
              </div>
            </form>
          </Panel>

          <Panel id="verdict" title="진단 판정" action={verdictStatus === undefined ? "대기" : statusLabel[verdictStatus]}>
            <VerdictPanel historyItem={selectedHistory} result={result} requestStatus={requestStatus} />
          </Panel>
        </section>

        <Panel id="evidence" title="진단 근거" action={trace?.trace_id ?? "대기"}>
          <EvidencePanel trace={trace} />
        </Panel>

        <Panel id="case-search" title="현재 점검 유사 사례" action={trace === null ? "대기" : `${currentSimilarCases.length}건`}>
          <CurrentSimilarCasesPanel cases={currentSimilarCases} hasTrace={trace !== null} requestStatus={requestStatus} />
        </Panel>

        <Panel id="detail" title="진단 상세" action={detail?.diagnosis.diagnosis_id ?? "선택 없음"}>
          <DetailPanel
            detail={detail}
            onDownload={() => void downloadReport()}
          />
        </Panel>

        <Panel id="report" title="운영 리포트" action={detail?.diagnosis.status === undefined ? "대기" : statusLabel[detail.diagnosis.status]}>
          <ReportPanel detail={detail} onDownload={() => void downloadReport()} />
        </Panel>

        <Panel id="rag-admin" title="LLM RAG 관리" action="문서·검색·재색인">
          <RagAdminPanel />
        </Panel>

        <section className="twoColumn">
          <Panel id="history" title="진단 이력" action={`${history.length}건`}>
            <DiagnosisTable items={history} onOpen={(item) => void openDiagnosis(item)} />
          </Panel>

          <Panel id="queue" title="검토 대기" action={`${reviewQueue.length}건`}>
            <ReviewQueue items={reviewQueue} onOpen={(item) => void openDiagnosis(item)} />
          </Panel>
        </section>

        <section id="trace" className="panel">
          <div className="panelHeader">
            <h3>처리 추적</h3>
            <span>{trace?.trace_id ?? "추적 없음"}</span>
          </div>
          <TraceLog expandedKey={expandedTraceKey} onToggle={setExpandedTraceKey} trace={trace} />
        </section>

        <Panel id="model-runtime" title="모델 런타임 상태" action={modelRuntimeAction(modelRuntime)}>
          <ModelRuntimePanel status={modelRuntime} />
        </Panel>

      </section>
    </main>
  );
}

function MetricCard(props: { readonly icon: ReactNode; readonly label: string; readonly value: string; readonly tone: string }) {
  return (
    <article className={`metricCard ${props.tone}`}>
      <div className="metricIcon">{props.icon}</div>
      <p>{props.label}</p>
      <strong>{props.value}</strong>
    </article>
  );
}

function Panel(props: { readonly id: string; readonly title: string; readonly action: string; readonly children: ReactNode }) {
  return (
    <section id={props.id} className="panel">
      <div className="panelHeader">
        <h3>{props.title}</h3>
        <span>{props.action}</span>
      </div>
      {props.children}
    </section>
  );
}

function UploadField(props: {
  readonly accept: string;
  readonly description: string;
  readonly icon: ReactNode;
  readonly label: string;
  readonly onChange: (file: File | null) => void;
}) {
  const inputId = useId();

  return (
    <div className="uploadField">
      <div className="uploadIcon">{props.icon}</div>
      <div>
        <strong>{props.label}</strong>
        <span>{props.description}</span>
      </div>
      <label className="uploadButton" htmlFor={inputId}>
        <UploadCloud size={16} />
        선택
      </label>
      <input
        accept={props.accept}
        className="fileInput"
        id={inputId}
        type="file"
        onClick={(event) => {
          event.currentTarget.value = "";
        }}
        onChange={(event) => props.onChange(event.currentTarget.files?.item(0) ?? null)}
      />
    </div>
  );
}

function InputInspector(props: {
  readonly csv: File | null;
  readonly image: File | null;
  readonly imagePreviewUrl: string | null;
}) {
  const imageStatus = validateImageFile(props.image);
  const csvStatus = validateCsvFile(props.csv);
  return (
    <div className="inputInspector">
      <article className="previewBox">
        <div className="previewHeader">
          <ImageIcon size={18} />
          <strong>PRPD 미리보기</strong>
          <span className={`miniStatus ${imageStatus.ok ? "ok" : "warn"}`}>{imageStatus.label}</span>
        </div>
        {props.imagePreviewUrl === null ? (
          <div className="previewEmpty">PNG 이미지를 선택하면 여기에 표시됩니다.</div>
        ) : (
          <img alt="선택된 PRPD 미리보기" src={props.imagePreviewUrl} />
        )}
      </article>
      <article className="validationBox">
        <div className="previewHeader">
          <FileCheck2 size={18} />
          <strong>CSV 입력 확인</strong>
          <span className={`miniStatus ${csvStatus.ok ? "ok" : "warn"}`}>{csvStatus.label}</span>
        </div>
        <dl>
          <dt>파일명</dt><dd>{props.csv?.name ?? "선택 없음"}</dd>
          <dt>크기</dt><dd>{props.csv === null ? "없음" : formatBytes(props.csv.size)}</dd>
          <dt>형식</dt><dd>{props.csv?.type || "없음"}</dd>
          <dt>상태</dt><dd>{csvStatus.message}</dd>
        </dl>
      </article>
    </div>
  );
}

function Field(props: { readonly label: string; readonly value: string; readonly onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input value={props.value} onChange={(event) => props.onChange(event.currentTarget.value)} />
    </label>
  );
}

function VerdictPanel(props: {
  readonly historyItem: DiagnosisListItem | null;
  readonly result: DiagnosisResponse | null;
  readonly requestStatus: RequestStatus;
}) {
  if (props.requestStatus === "failed") {
    return (
      <div className="emptyState error">
        <AlertTriangle size={28} />
        <strong>요청 실패</strong>
        <span>서버 상태 또는 업로드 파일을 확인해 주세요.</span>
      </div>
    );
  }
  if (props.result === null) {
    if (props.historyItem !== null) {
      return <HistoryVerdict item={props.historyItem} />;
    }
    return (
      <div className="emptyState">
        <ShieldCheck size={28} />
        <strong>진단 대기</strong>
        <span>진단을 실행하면 최종 판정이 표시됩니다.</span>
      </div>
    );
  }

  return (
    <div className="verdictBox">
      <div className={`status ${props.result.status}`}>{statusLabel[props.result.status]}</div>
      <h4>{props.result.diagnosis ?? "판정 보류"}</h4>
      <dl>
        <dt>진단 ID</dt><dd>{props.result.diagnosis_id}</dd>
        <dt>위험도</dt><dd>{props.result.risk_level ?? "없음"}</dd>
        <dt>신뢰도</dt><dd>{props.result.confidence === null ? "없음" : `${Math.round(props.result.confidence * 100)}%`}</dd>
        <dt>경로</dt><dd>{routeLabel[props.result.route]}</dd>
      </dl>
      <div className="reasonBox">
        <strong>판정 근거</strong>
        <p>{props.result.reason}</p>
      </div>
      <div className="reasonBox">
        <strong>권고 조치</strong>
        <p>{props.result.recommended_action ?? "운영자 검토가 필요합니다."}</p>
      </div>
    </div>
  );
}

function HistoryVerdict(props: { readonly item: DiagnosisListItem }) {
  return (
    <div className="verdictBox">
      <div className={`status ${props.item.status}`}>{statusLabel[props.item.status]}</div>
      <h4>{props.item.diagnosis ?? "판정 보류"}</h4>
      <dl>
        <dt>진단 ID</dt><dd>{props.item.diagnosis_id}</dd>
        <dt>위험도</dt><dd>{props.item.risk_level ?? "없음"}</dd>
        <dt>신뢰도</dt><dd>{props.item.confidence === null ? "없음" : `${Math.round(props.item.confidence * 100)}%`}</dd>
        <dt>경로</dt><dd>{routeLabel[props.item.route]}</dd>
      </dl>
      <div className="reasonBox">
        <strong>판정 근거</strong>
        <p>{props.item.reason}</p>
      </div>
    </div>
  );
}

function DetailPanel(props: {
  readonly detail: DiagnosisDetailResponse | null;
  readonly onDownload: () => void;
}) {
  if (props.detail === null) {
    return (
      <div className="emptyState compact">
        <FileCheck2 size={24} />
        <strong>선택된 진단 없음</strong>
        <span>진단을 실행하거나 이력 항목을 열면 운영 기록을 확인할 수 있습니다.</span>
      </div>
    );
  }
  const referenceCases = similarCasesFromTrace(props.detail.trace);
  const timeline = props.detail.timeline.filter((event) => event.kind !== "action" && event.kind !== "comment");
  return (
    <div className="detailGrid">
      <section className="detailSummary">
        <div>
          <p className="eyebrow">선택 진단</p>
          <h4>{props.detail.diagnosis.diagnosis_id}</h4>
        </div>
        <button className="downloadButton" type="button" onClick={props.onDownload}>
          <Download size={16} />
          보고서 다운로드
        </button>
      </section>

      {referenceCases.length > 0 ? (
        <SimilarCaseBoard cases={referenceCases} title="리포트 참조 사례" />
      ) : null}

      <section className="caseTimeline">
        <div className="sectionTitle">케이스 타임라인</div>
        <div className="timelineList">
          {timeline.map((event) => (
            <article className={`timelineEvent ${event.kind}`} key={`${event.created_at}-${event.kind}-${event.title}`}>
              <div>
                <strong>{event.title}</strong>
                <span>{timelineKindLabel(event.kind)} · {formatDate(event.created_at)}</span>
              </div>
              <p>{event.body}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ReportPanel(props: {
  readonly detail: DiagnosisDetailResponse | null;
  readonly onDownload: () => void;
}) {
  if (props.detail === null) {
    return (
      <div className="emptyState compact">
        <FileSearch size={24} />
        <strong>리포트 대기</strong>
        <span>진단 상세가 선택되면 관리자 검토용 리포트가 구성됩니다.</span>
      </div>
    );
  }

  const signals = modelSignalsFromTrace(props.detail.trace);
  const ragDocuments = ragDocumentsFromTrace(props.detail.trace);
  const referenceCases = similarCasesFromTrace(props.detail.trace);
  const fusion = parseFusionSummary(findTraceEvent(props.detail.trace, "fusion_engine")?.summary);
  return (
    <div className="reportCanvas">
      <section className="reportHero">
        <div>
          <p className="eyebrow">관리자 검토 리포트</p>
          <h4>{props.detail.diagnosis.diagnosis ?? "판정 보류"}</h4>
          <p>{props.detail.diagnosis.reason}</p>
        </div>
        <div className="reportActions">
          <button className="downloadButton" type="button" onClick={props.onDownload}>
            <Download size={16} />
            JSON
          </button>
          <button className="downloadButton" type="button" onClick={() => window.print()}>
            <Printer size={16} />
            인쇄/PDF
          </button>
        </div>
      </section>

      <section className="reportMetricGrid">
        <RuntimeStatusItem label="상태" value={statusLabel[props.detail.diagnosis.status]} tone={props.detail.diagnosis.status === "completed" ? "ready" : "warn"} />
        <RuntimeStatusItem label="위험도" value={props.detail.diagnosis.risk_level ?? "없음"} tone="neutral" />
        <RuntimeStatusItem label="신뢰도" value={formatConfidence(props.detail.diagnosis.confidence)} tone="neutral" />
      </section>

      <section className="reportSection">
        <div className="sectionTitle">모델 근거 요약</div>
        <div className="reportSignalGrid">
          {signals.map((signal) => (
            <article className="reportSignalCard" key={signal.source}>
              <span>{signal.source}</span>
              <strong>{signal.label}</strong>
              <p>{signal.detail}</p>
              <b>{formatConfidence(signal.confidence)}</b>
            </article>
          ))}
        </div>
        {fusion !== null ? <p className="reportNote">{agreementLabel(fusion.agreement_level)} · {fusion.rationale}</p> : null}
      </section>

      <section className="reportSection">
        <div className="sectionTitle">RAG 근거 문서</div>
        <div className="reportDocumentList">
          {ragDocuments.length === 0 ? (
            <div className="similarCaseEmpty">
              <BookOpen size={20} />
              <span>리포트에 포함된 RAG 문서가 없습니다.</span>
            </div>
          ) : ragDocuments.slice(0, 4).map((document) => (
            <article className="ragDocumentCard" key={document.document_id}>
              <span>{sourceTypeLabel(document.source_type ?? undefined)}</span>
              <strong>{document.title}</strong>
              <p>{document.excerpt}</p>
              <small>{document.source} · 관련도 {formatMetric(document.relevance)}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="reportSection">
        <div className="sectionTitle">참조 사례 및 운영 기록</div>
        <div className="reportSplit">
          <SimilarCaseBoard cases={referenceCases.slice(0, 2)} title="리포트 참조 사례" />
          <div className="operationLog">
            <strong>조치/메모</strong>
            {[...props.detail.actions, ...props.detail.comments].length === 0 ? (
              <p>등록된 관리자 조치나 코멘트가 없습니다.</p>
            ) : (
              [...props.detail.actions, ...props.detail.comments].map((item, index) => (
                <p key={`${item.created_at}-${index}`}>
                  <span>{formatDate(item.created_at)}</span>
                  {operationRecordLabel(item)} · {item.note}
                </p>
              ))
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function RagAdminPanel() {
  const [panelStatus, setPanelStatus] = useState<RagPanelStatus>("loading");
  const [status, setStatus] = useState<RagStatusResponse | null>(null);
  const [documents, setDocuments] = useState<readonly RagDocumentListItem[]>([]);
  const [queryLogs, setQueryLogs] = useState<readonly RagQueryLogItem[]>([]);
  const [sourceType, setSourceType] = useState("all");
  const [searchQuery, setSearchQuery] = useState("HFCT 코로나 방전 근거");
  const [searchTopK, setSearchTopK] = useState(6);
  const [searchResults, setSearchResults] = useState<readonly RagDocument[]>([]);
  const [datasetLimit, setDatasetLimit] = useState("");
  const [reindexResult, setReindexResult] = useState<RagReindexResponse | null>(null);

  useEffect(() => {
    void refreshRagAdmin();
  }, [sourceType]);

  async function refreshRagAdmin(): Promise<void> {
    setPanelStatus("loading");
    try {
      const [nextStatus, nextDocuments, nextLogs] = await Promise.all([
        fetchRagStatus(),
        fetchRagDocuments({sourceType, limit: RAG_ADMIN_LIST_LIMIT}),
        fetchRagQueryLogs(RAG_ADMIN_LIST_LIMIT),
      ]);
      setStatus(nextStatus);
      setDocuments(nextDocuments.items.slice(0, RAG_ADMIN_LIST_LIMIT));
      setQueryLogs(nextLogs.items.slice(0, RAG_ADMIN_LIST_LIMIT));
      setPanelStatus("idle");
    } catch {
      setPanelStatus("failed");
    }
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const query = searchQuery.trim();
    if (query.length === 0) {
      return;
    }
    setPanelStatus("searching");
    try {
      const result = await searchRagDocuments({query, topK: boundedRagTopK(searchTopK)});
      setSearchResults(result.documents);
      setPanelStatus("idle");
    } catch {
      setPanelStatus("failed");
    }
  }

  async function handleReindex(): Promise<void> {
    setPanelStatus("reindexing");
    try {
      const result = await reindexRagDocuments(parseDatasetLimit(datasetLimit));
      setReindexResult(result);
      await refreshRagAdmin();
    } catch {
      setPanelStatus("failed");
    }
  }

  const sourceTypes = ragSourceTypes(status);
  return (
    <div className="ragAdminPanel">
      <section className="ragStatusGrid">
        <RuntimeStatusItem label="RAG 상태" value={status === null ? "확인 중" : status.ready ? "준비됨" : "확인 필요"} tone={status?.ready ? "ready" : "warn"} />
        <RuntimeStatusItem label="문서" value={status === null ? "0" : status.document_count.toString()} tone="neutral" />
        <RuntimeStatusItem label="Chunk" value={status === null ? "0" : status.chunk_count.toString()} tone="neutral" />
        <RuntimeStatusItem label="Query log" value={status === null ? "0" : status.query_log_count.toString()} tone="neutral" />
      </section>

      <section className="ragControlBar">
        <SourceTypeTabs selected={sourceType} sourceTypes={sourceTypes} status={status} onSelect={setSourceType} />
        <label className="field">
          <span>Dataset limit</span>
          <input inputMode="numeric" max={50000} min={1} placeholder="전체" value={datasetLimit} onChange={(event) => setDatasetLimit(event.currentTarget.value)} />
        </label>
        <button className="downloadButton" type="button" onClick={() => void refreshRagAdmin()}>
          <RefreshCw size={16} />
          새로고침
        </button>
        <button className="primaryButton" type="button" disabled={panelStatus === "reindexing"} onClick={() => void handleReindex()}>
          {panelStatus === "reindexing" ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
          재색인
        </button>
      </section>

      {status?.error ? <p className="ragError">{status.error}</p> : null}
      {reindexResult !== null ? (
        <p className="ragNotice">
          재색인 완료: 문서 {reindexResult.document_count}건, chunk {reindexResult.chunk_count}건, embedding {reindexResult.embedding_model}
        </p>
      ) : null}

      <form className="ragSearchForm" onSubmit={(event) => void handleSearch(event)}>
        <label className="field">
          <span>RAG 검색 테스트</span>
          <input value={searchQuery} onChange={(event) => setSearchQuery(event.currentTarget.value)} />
        </label>
        <TopKControl value={searchTopK} onChange={setSearchTopK} />
        <button className="primaryButton" type="submit" disabled={panelStatus === "searching"}>
          {panelStatus === "searching" ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
          검색
        </button>
      </form>

      <section className="ragAdminGrid">
        <RagSearchResults documents={searchResults} />
        <RagDocumentTable documents={documents} />
        <RagQueryLogList logs={queryLogs} />
      </section>

      {panelStatus === "failed" ? (
        <div className="similarCaseEmpty ragError">
          <AlertTriangle size={20} />
          <span>RAG 관리 API 호출에 실패했습니다.</span>
        </div>
      ) : null}
    </div>
  );
}

function SourceTypeTabs(props: {
  readonly onSelect: (sourceType: string) => void;
  readonly selected: string;
  readonly sourceTypes: readonly string[];
  readonly status: RagStatusResponse | null;
}) {
  const options = ["all", ...props.sourceTypes];
  return (
    <div className="sourceTypePanel">
      <span>Source type</span>
      <div className="sourceTypeTabs" role="tablist" aria-label="RAG source type">
        {options.map((sourceType) => (
          <button
            aria-selected={props.selected === sourceType}
            className={`sourceTypeButton ${props.selected === sourceType ? "active" : ""}`}
            key={sourceType}
            role="tab"
            type="button"
            onClick={() => props.onSelect(sourceType)}
          >
            <strong>{sourceType === "all" ? "전체" : sourceTypeLabel(sourceType)}</strong>
            <small>{sourceTypeCount(sourceType, props.status).toLocaleString()} docs</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function TopKControl(props: {
  readonly onChange: (value: number) => void;
  readonly value: number;
}) {
  const value = boundedRagTopK(props.value);
  return (
    <div className="topKControl">
      <span>Top K</span>
      <div>
        <button
          aria-label="Top K 감소"
          disabled={value <= RAG_TOP_K_MIN}
          type="button"
          onClick={() => props.onChange(boundedRagTopK(value - 1))}
        >
          -
        </button>
        <input
          aria-label="Top K"
          inputMode="numeric"
          max={RAG_TOP_K_MAX}
          min={RAG_TOP_K_MIN}
          type="number"
          value={value}
          onChange={(event) => props.onChange(boundedRagTopK(Number(event.currentTarget.value)))}
        />
        <button
          aria-label="Top K 증가"
          disabled={value >= RAG_TOP_K_MAX}
          type="button"
          onClick={() => props.onChange(boundedRagTopK(value + 1))}
        >
          +
        </button>
      </div>
    </div>
  );
}

function RagSearchResults(props: { readonly documents: readonly RagDocument[] }) {
  return (
    <section className="ragAdminCard">
      <div className="sectionTitle">검색 결과</div>
      {props.documents.length === 0 ? (
        <div className="similarCaseEmpty">
          <Search size={20} />
          <span>검색을 실행하면 상위 문서가 표시됩니다.</span>
        </div>
      ) : (
        <div className="reportDocumentList">
          {props.documents.map((document) => (
            <article className="ragDocumentCard" key={document.document_id}>
              <span>{sourceTypeLabel(document.source_type ?? undefined)}</span>
              <strong>{document.title}</strong>
              <p>{document.excerpt}</p>
              <small>{document.source} · 관련도 {formatMetric(document.relevance)}</small>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function RagDocumentTable(props: { readonly documents: readonly RagDocumentListItem[] }) {
  return (
    <section className="ragAdminCard">
      <div className="sectionTitle">문서 목록 TOP 5</div>
      <div className="compactTable">
        {props.documents.map((document) => (
          <article className="compactRow" key={document.document_key}>
            <div>
              <strong>{document.title}</strong>
              <span>{document.source_path ?? document.document_key}</span>
            </div>
            <b>{sourceTypeLabel(document.source_type)}</b>
            <small>{document.chunk_count} chunks</small>
          </article>
        ))}
        {props.documents.length === 0 ? (
          <div className="similarCaseEmpty">
            <BookOpen size={20} />
            <span>등록된 RAG 문서가 없습니다.</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function RagQueryLogList(props: { readonly logs: readonly RagQueryLogItem[] }) {
  return (
    <section className="ragAdminCard">
      <div className="sectionTitle">질의 로그 TOP 5</div>
      <div className="compactTable">
        {props.logs.map((log) => (
          <article className="compactRow" key={log.id}>
            <div>
              <strong>{log.query_text}</strong>
              <span>{log.diagnosis_id ?? "standalone"} · {formatDate(log.created_at)}</span>
            </div>
            <small>{log.retrieved_chunks.length} chunks</small>
          </article>
        ))}
        {props.logs.length === 0 ? (
          <div className="similarCaseEmpty">
            <FileText size={20} />
            <span>저장된 RAG 질의 로그가 없습니다.</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function DiagnosisTable(props: {
  readonly items: readonly DiagnosisListItem[];
  readonly onOpen: (item: DiagnosisListItem) => void;
}) {
  if (props.items.length === 0) {
    return (
      <div className="emptyState compact">
        <Database size={24} />
        <strong>진단 이력 없음</strong>
        <span>완료된 점검 결과가 이곳에 표시됩니다.</span>
      </div>
    );
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>진단 ID</th>
            <th>상태</th>
            <th>경로</th>
            <th>위험도</th>
            <th>신뢰도</th>
            <th>생성 시각</th>
          </tr>
        </thead>
        <tbody>
          {props.items.map((item) => (
            <tr className="clickableRow" key={item.diagnosis_id} onClick={() => props.onOpen(item)}>
              <td><strong>{item.diagnosis_id}</strong></td>
              <td><span className={`status ${item.status}`}>{statusLabel[item.status]}</span></td>
              <td>{routeLabel[item.route]}</td>
              <td>{item.risk_level ?? "없음"}</td>
              <td>{formatConfidence(item.confidence)}</td>
              <td>{formatDate(item.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReviewQueue(props: {
  readonly items: readonly DiagnosisListItem[];
  readonly onOpen: (item: DiagnosisListItem) => void;
}) {
  if (props.items.length === 0) {
    return (
      <div className="emptyState compact">
        <ShieldCheck size={24} />
        <strong>검토 대기 없음</strong>
        <span>반려되었거나 검토가 필요한 진단이 없습니다.</span>
      </div>
    );
  }

  return (
    <div className="coverageList">
      {props.items.map((item) => (
        <button className="queueRow" key={item.diagnosis_id} type="button" onClick={() => props.onOpen(item)}>
          <div>
            <strong>{item.diagnosis ?? item.diagnosis_id}</strong>
            <span>{item.reason}</span>
          </div>
          <span className={`status ${item.status}`}>{statusLabel[item.status]}</span>
        </button>
      ))}
    </div>
  );
}

function TraceLog(props: {
  readonly expandedKey: string | null;
  readonly onToggle: (key: string | null) => void;
  readonly trace: TraceResponse | null;
}) {
  if (props.trace === null) {
    return (
      <div className="emptyState compact">
        <Cpu size={24} />
        <strong>처리 추적 없음</strong>
        <span>라우팅, 모델, 검토, 리포트 생성 단계가 이곳에 표시됩니다.</span>
      </div>
    );
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>순서</th>
            <th>유형</th>
            <th>단계</th>
          </tr>
        </thead>
        <tbody>
          {props.trace.events.map((event, index) => (
            <TraceRow
              expanded={props.expandedKey === traceEventKey(event.kind, event.name, index)}
              index={index}
              key={traceEventKey(event.kind, event.name, index)}
              kind={event.kind}
              name={event.name}
              onToggle={props.onToggle}
              summary={event.summary}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TraceRow(props: {
  readonly expanded: boolean;
  readonly index: number;
  readonly kind: string;
  readonly name: string;
  readonly onToggle: (key: string | null) => void;
  readonly summary: Record<string, unknown>;
}) {
  const key = traceEventKey(props.kind, props.name, props.index);
  return (
    <>
      <tr className="clickableRow" onClick={() => props.onToggle(props.expanded ? null : key)}>
        <td>{props.index + 1}</td>
        <td>{props.kind}</td>
        <td><strong>{props.name}</strong></td>
      </tr>
      {props.expanded ? (
        <tr>
          <td colSpan={3}>
            <pre className="traceSummary">{JSON.stringify(props.summary, null, 2)}</pre>
          </td>
        </tr>
      ) : null}
    </>
  );
}

function ModelRuntimePanel(props: { readonly status: ModelRuntimeStatus | null }) {
  if (props.status === null) {
    return (
      <div className="emptyState compact">
        <Cpu size={24} />
        <strong>모델 상태 확인 중</strong>
        <span>백엔드 런타임에서 모델 adapter 상태를 가져오고 있습니다.</span>
      </div>
    );
  }

  return (
    <div className="runtimePanel">
      <div className="runtimeSummary">
        <RuntimeStatusItem label="Adapter 모드" value={props.status.adapter_mode} tone="neutral" />
        <RuntimeStatusItem label="Agents SDK" value={props.status.agents_sdk_installed ? "설치됨" : "미설치"} tone={props.status.agents_sdk_installed ? "ready" : "warn"} />
        <RuntimeStatusItem label="LLM RAG" value={props.status.llm_rag_ready ? "OpenRouter 연결" : "Fallback"} tone={props.status.llm_rag_ready ? "ready" : "warn"} />
      </div>
      <div className="runtimeGrid">
        <RuntimeModelCard
          adapter={props.status.time_series_adapter}
          checkpoint={props.status.time_series_checkpoint}
          error={props.status.time_series_error}
          manifest={props.status.time_series_manifest}
          model={props.status.time_series_model}
          ready={props.status.time_series_ready}
          title="시계열 모델"
          version={props.status.time_series_version}
        />
        <RuntimeModelCard
          adapter={props.status.vision_adapter}
          checkpoint={props.status.vision_checkpoint}
          error={props.status.vision_error}
          manifest={props.status.vision_manifest}
          model={props.status.vision_model}
          ready={props.status.vision_ready}
          title="비전 모델"
          version={props.status.vision_version}
        />
        <RuntimeModelCard
          adapter={props.status.vlm_adapter}
          checkpoint={props.status.vlm_checkpoint}
          error={props.status.vlm_error}
          manifest={props.status.vlm_manifest}
          model={props.status.vlm_model}
          ready={props.status.vlm_ready}
          title="VLM 모델"
          version={props.status.vlm_version}
        />
        <RuntimeModelCard
          adapter={props.status.llm_rag_adapter}
          checkpoint={null}
          error={props.status.llm_rag_error}
          manifest={null}
          model={props.status.llm_rag_model ?? "미설정"}
          ready={props.status.llm_rag_ready}
          title="LLM RAG"
          version={props.status.llm_rag_provider}
        />
      </div>
      <div className="runtimeFootnote">
        <span>RAG</span>
        <strong>{props.status.rag_retriever}</strong>
        <span>{props.status.rag_version}</span>
      </div>
      <div className="runtimeFootnote">
        <span>Artifact root</span>
        <strong>{props.status.artifact_root}</strong>
      </div>
    </div>
  );
}

function RuntimeStatusItem(props: {
  readonly label: string;
  readonly tone: "neutral" | "ready" | "warn";
  readonly value: string;
}) {
  return (
    <article className={`runtimeStatusItem ${props.tone}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </article>
  );
}

function RuntimeModelCard(props: {
  readonly adapter: string;
  readonly checkpoint: string | null;
  readonly error: string | null;
  readonly manifest: string | null;
  readonly model: string;
  readonly ready: boolean;
  readonly title: string;
  readonly version: string;
}) {
  return (
    <article className="runtimeModelCard">
      <div className="runtimeModelHeader">
        <span className={`miniStatus ${props.ready ? "ok" : "warn"}`}>{props.ready ? "준비됨" : "확인 필요"}</span>
        <strong>{props.title}</strong>
      </div>
      <dl>
        <dt>모델</dt><dd>{props.model}</dd>
        <dt>버전</dt><dd>{props.version}</dd>
        <dt>Adapter</dt><dd>{props.adapter}</dd>
        <dt>Manifest</dt><dd>{props.manifest ?? "없음"}</dd>
        <dt>Checkpoint</dt><dd>{props.checkpoint ?? "없음"}</dd>
        <dt>상태</dt><dd>{props.error ?? "정상"}</dd>
      </dl>
    </article>
  );
}

function EvidencePanel(props: { readonly trace: TraceResponse | null }) {
  if (props.trace === null) {
    return (
      <div className="emptyState compact">
        <BarChart3 size={24} />
        <strong>진단 근거 없음</strong>
        <span>진단을 실행하면 설비 정보, 시계열, 비전, 유사 사례, 지식 검색, VLM 결과가 표시됩니다.</span>
      </div>
    );
  }
  const inputArtifacts = inputArtifactsFromTrace(props.trace);
  const modelSignals = modelSignalsFromTrace(props.trace);
  const rows = [
    { title: "설비 정보", event: findTraceEvent(props.trace, "metadata_context") },
    { title: "시계열", event: findTraceEvent(props.trace, "time_series_tool") },
    { title: "비전", event: findTraceEvent(props.trace, "vision_tool") },
    { title: "유사 사례", event: findTraceEvent(props.trace, "similar_case_tool") },
    { title: "지식 검색", event: findTraceEvent(props.trace, "rag_tool") },
    { title: "VLM 리포트", event: findTraceEvent(props.trace, "vlm_tool") },
    { title: "융합 판단", event: findTraceEvent(props.trace, "fusion_engine") },
  ];
  const similarCases = similarCasesFromTrace(props.trace);
  return (
    <div className="evidenceStack">
      <section className="evidenceWorkbench">
        <PrpdEvidenceViewer imageUrl={inputArtifacts.prpdImageUrl} similarCases={similarCases} />
        <SignalEvidencePanel summary={inputArtifacts.signalSummary} trace={props.trace} />
        <ModelAgreementPanel signals={modelSignals} />
      </section>
      <div className="evidenceGrid">
        {rows.map((row) => (
          <article className={`evidenceCard ${row.event === undefined ? "pending" : ""}`} key={row.title}>
            <div className="evidenceHeader">
              <span>{row.title}</span>
              <strong>{evidenceSourceTitle(row.event)}</strong>
            </div>
            <dl>
              {evidenceMetricRows(row.event).map((metric) => (
                <Fragment key={metric.label}>
                  <dt>{metric.label}</dt><dd>{metric.value}</dd>
                </Fragment>
              ))}
            </dl>
            {evidencePayload(row.event) !== null ? (
              <pre className="evidenceSummary">{JSON.stringify(evidencePayload(row.event), null, 2)}</pre>
            ) : null}
          </article>
        ))}
      </div>
      <SimilarCaseBoard cases={similarCases} title="데이터셋 유사 사례" />
    </div>
  );
}

function PrpdEvidenceViewer(props: {
  readonly imageUrl: string | null;
  readonly similarCases: readonly SimilarCase[];
}) {
  return (
    <article className="workbenchCard prpdCard">
      <div className="workbenchHeader">
        <span>PRPD 이미지</span>
        <strong>{props.imageUrl === null ? "입력 이미지 없음" : "위상 기준 확대"}</strong>
      </div>
      <div className="prpdViewport">
        {props.imageUrl === null ? (
          <div className="previewEmpty">현재 진단 trace에 PRPD 이미지가 없습니다.</div>
        ) : (
          <>
            <img alt="진단 PRPD 이미지" src={apiAssetUrl(props.imageUrl)} />
            <div className="phaseOverlay" aria-hidden="true">
              <span>0</span>
              <span>90</span>
              <span>180</span>
              <span>270</span>
              <span>360</span>
            </div>
          </>
        )}
      </div>
      <div className="sideBySideStrip">
        <div>
          <span>현재</span>
          {props.imageUrl === null ? null : <img alt="현재 PRPD 비교 이미지" src={apiAssetUrl(props.imageUrl)} />}
          <strong>{props.imageUrl === null ? "없음" : "점검 이미지"}</strong>
        </div>
        {props.similarCases.slice(0, 2).map((item) => (
          <div key={item.sample_id}>
            <span>{formatMetric(item.similarity)}</span>
            <img alt={`${item.sample_id} 유사 사례`} src={apiAssetUrl(item.image_url)} />
            <strong>{item.label_name}</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function SignalEvidencePanel(props: {
  readonly summary: SignalSummary | null;
  readonly trace: TraceResponse;
}) {
  const features = timeseriesFeaturesFromTrace(props.trace);
  const peakAbs = props.summary?.peak_abs ?? numericRecordValue(features, "peak_abs") ?? numericRecordValue(features, "abs_peak");
  const rms = props.summary?.rms ?? numericRecordValue(features, "rms");
  const p99 = props.summary?.p99_abs ?? numericRecordValue(features, "abs_p99");
  return (
    <article className="workbenchCard signalCard">
      <div className="workbenchHeader">
        <span>시계열 CSV</span>
        <strong>{props.summary === null ? "모델 feature 기준" : `${props.summary.sample_count.toLocaleString()} samples`}</strong>
      </div>
      <div className="signalMetricGrid">
        <SignalMetric label="Peak" value={formatNumeric(peakAbs)} />
        <SignalMetric label="RMS" value={formatNumeric(rms)} />
        <SignalMetric label="P99" value={formatNumeric(p99)} />
        <SignalMetric label="이상률" value={props.summary === null ? "없음" : formatMetric(props.summary.anomaly_rate)} />
      </div>
      <div className="anomalyRail" aria-label="이상 구간 요약">
        {props.summary?.anomaly_regions.length ? (
          props.summary.anomaly_regions.map((region) => (
            <span
              key={`${region.frame}-${region.start_index}-${region.end_index}`}
              style={{
                left: `${phasePosition(region.start_index, props.summary?.channel_count ?? 1)}%`,
                width: `${phaseWidth(region.start_index, region.end_index, props.summary?.channel_count ?? 1)}%`,
              }}
              title={`frame ${region.frame}, peak ${formatNumeric(region.peak_abs)}`}
            />
          ))
        ) : null}
      </div>
      <div className="signalFootnote">
        <span>Threshold {props.summary === null ? "없음" : formatNumeric(props.summary.anomaly_threshold)}</span>
        <span>Mean {props.summary === null ? "없음" : formatNumeric(props.summary.mean)}</span>
      </div>
    </article>
  );
}

function SignalMetric(props: { readonly label: string; readonly value: string }) {
  return (
    <div className="signalMetric">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function ModelAgreementPanel(props: { readonly signals: readonly ModelSignal[] }) {
  const conflict = modelConflict(props.signals);
  return (
    <article className={`workbenchCard agreementCard ${conflict ? "conflict" : "aligned"}`}>
      <div className="workbenchHeader">
        <span>모델 비교</span>
        <strong>{conflict ? "판정 불일치" : "판정 정합"}</strong>
      </div>
      <div className="modelSignalList">
        {props.signals.map((signal) => (
          <div className="modelSignalRow" key={signal.source}>
            <span>{signal.source}</span>
            <strong>{signal.label}</strong>
            <b>{formatConfidence(signal.confidence)}</b>
            <small>{signal.detail}</small>
          </div>
        ))}
      </div>
    </article>
  );
}

function CurrentSimilarCasesPanel(props: {
  readonly cases: readonly SimilarCase[];
  readonly hasTrace: boolean;
  readonly requestStatus: RequestStatus;
}) {
  if (!props.hasTrace) {
    return (
      <section className="currentCasePanel idle">
        <div className="currentCaseBrief">
          <span>자동 매칭</span>
          <strong>현재 점검 기준 대기</strong>
        </div>
        <div className="similarCaseEmpty">
          {props.requestStatus === "submitting" ? <Loader2 className="spin" size={22} /> : <ImageIcon size={22} />}
          <span>{props.requestStatus === "submitting" ? "과거 사례 대조 중입니다." : "현재 점검 결과가 아직 없습니다."}</span>
        </div>
      </section>
    );
  }

  return (
    <div className="currentCasePanel">
      <div className="currentCaseBrief">
        <span>자동 매칭</span>
        <strong>{props.cases.length > 0 ? "현재 점검과 가장 가까운 과거 사례" : "매칭된 과거 사례 없음"}</strong>
      </div>
      <SimilarCaseBoard cases={props.cases} title="현재 점검 유사 사례" hideHeader />
    </div>
  );
}

function SimilarCaseBoard(props: { readonly cases: readonly SimilarCase[]; readonly hideHeader?: boolean; readonly title: string }) {
  if (props.cases.length === 0) {
    return (
      <section className="similarCaseSection empty">
        {props.hideHeader === true ? null : <div className="sectionTitle">{props.title}</div>}
        <div className="similarCaseEmpty">
          <ImageIcon size={22} />
          <span>표시할 데이터셋 참조 사례가 없습니다.</span>
        </div>
      </section>
    );
  }
  const topCase = props.cases[0];
  if (topCase === undefined) {
    return null;
  }

  return (
    <section className="similarCaseSection">
      {props.hideHeader === true ? null : (
        <div className="similarCaseHeader">
          <div>
            <div className="sectionTitle">{props.title}</div>
            <strong>상위 {props.cases.length}건 현장 참조</strong>
          </div>
          <span>최고 유사도 {formatMetric(topCase.similarity)}</span>
        </div>
      )}
      <div className="similarCaseGrid">
        {props.cases.map((item) => (
          <article className="similarCaseCard" key={item.sample_id}>
            <div className="caseThumb">
              <img alt={`${item.label_name} PRPD 참조 이미지`} loading="lazy" src={apiAssetUrl(item.image_url)} />
            </div>
            <div className="caseBody">
              <div className="caseTitle">
                <span className="status completed">{item.label_name}</span>
                <strong>{item.sample_id}</strong>
              </div>
              <div className="caseScore" aria-label={`유사도 ${formatMetric(item.similarity)}`}>
                <span style={{width: `${similarityWidth(item.similarity)}%`}} />
              </div>
              <dl>
                <dt>유사도</dt><dd>{formatMetric(item.similarity)}</dd>
                <dt>설비</dt><dd>{item.equipment_name || "없음"}</dd>
                <dt>센서</dt><dd>{item.sensor_type || "없음"}</dd>
                <dt>절연</dt><dd>{item.insulator_type || "없음"}</dd>
                <dt>이격</dt><dd>{item.clearance_distance || "없음"}</dd>
                <dt>최대값</dt><dd>{metadataValue(item, "max_discharge_value")}</dd>
              </dl>
              <p>{item.reason}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function findTraceEvent(trace: TraceResponse, name: string): TraceResponse["events"][number] | undefined {
  return trace.events.find((event) => event.name === name);
}

function similarCasesFromTrace(trace: TraceResponse | null): readonly SimilarCase[] {
  if (trace === null) {
    return [];
  }
  return parseSimilarCases(findTraceEvent(trace, "similar_case_tool")?.summary["cases"]);
}

function inputArtifactsFromTrace(trace: TraceResponse): InputArtifactEvidence {
  const summary = findTraceEvent(trace, "input_artifacts")?.summary;
  return {
    prpdImageUrl: stringValue(summary?.["prpd_image_url"]) ?? null,
    timeseriesCsvUrl: stringValue(summary?.["timeseries_csv_url"]) ?? null,
    signalSummary: parseSignalSummary(summary?.["timeseries_signal"]),
  };
}

function parseSignalSummary(value: unknown): SignalSummary | null {
  const record = asRecord(value);
  if (record === null) {
    return null;
  }
  const anomalyRegions = parseSignalAnomalyRegions(record["anomaly_regions"]);
  if (
    typeof record["frame_count"] !== "number" ||
    typeof record["channel_count"] !== "number" ||
    typeof record["sample_count"] !== "number" ||
    typeof record["mean"] !== "number" ||
    typeof record["rms"] !== "number" ||
    typeof record["peak_abs"] !== "number" ||
    typeof record["p99_abs"] !== "number" ||
    typeof record["anomaly_threshold"] !== "number" ||
    typeof record["anomaly_count"] !== "number" ||
    typeof record["anomaly_rate"] !== "number" ||
    anomalyRegions === null
  ) {
    return null;
  }
  return {
    frame_count: record["frame_count"],
    channel_count: record["channel_count"],
    sample_count: record["sample_count"],
    mean: record["mean"],
    rms: record["rms"],
    peak_abs: record["peak_abs"],
    p99_abs: record["p99_abs"],
    anomaly_threshold: record["anomaly_threshold"],
    anomaly_count: record["anomaly_count"],
    anomaly_rate: record["anomaly_rate"],
    anomaly_regions: anomalyRegions,
  };
}

function parseSignalAnomalyRegions(value: unknown): readonly SignalAnomalyRegion[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const regions = value.filter(isSignalAnomalyRegion);
  return regions.length === value.length ? regions : null;
}

function isSignalAnomalyRegion(value: unknown): value is SignalAnomalyRegion {
  const record = asRecord(value);
  return (
    record !== null &&
    typeof record["frame"] === "number" &&
    typeof record["start_index"] === "number" &&
    typeof record["end_index"] === "number" &&
    typeof record["count"] === "number" &&
    typeof record["peak_abs"] === "number"
  );
}

function timeseriesFeaturesFromTrace(trace: TraceResponse): Record<string, unknown> {
  return asRecord(findTraceEvent(trace, "time_series_tool")?.summary["features"]) ?? {};
}

function modelSignalsFromTrace(trace: TraceResponse): readonly ModelSignal[] {
  return [
    modelSignalFromEvent("시계열", findTraceEvent(trace, "time_series_tool")),
    modelSignalFromEvent("비전", findTraceEvent(trace, "vision_tool")),
    modelSignalFromEvent("VLM", findTraceEvent(trace, "vlm_tool")),
    similarCaseSignal(findTraceEvent(trace, "similar_case_tool")),
    ragSignal(findTraceEvent(trace, "rag_tool")),
    fusionSignal(findTraceEvent(trace, "fusion_engine")),
  ].filter((signal): signal is ModelSignal => signal !== null);
}

function modelSignalFromEvent(source: string, event: TraceResponse["events"][number] | undefined): ModelSignal | null {
  if (event === undefined) {
    return null;
  }
  const evidence = parseStandardEvidence(event.summary["standard_evidence"]);
  return {
    source,
    label: evidence?.label_name ?? summaryValue(event, "label_name"),
    confidence: evidence?.confidence ?? numericSummaryValue(event, "confidence"),
    detail: evidence?.explanation ?? evidenceSourceTitle(event),
  };
}

function similarCaseSignal(event: TraceResponse["events"][number] | undefined): ModelSignal | null {
  if (event === undefined) {
    return null;
  }
  return {
    source: "유사 사례",
    label: summaryValue(event, "top_label"),
    confidence: numericSummaryValue(event, "top_similarity"),
    detail: `${summaryValue(event, "case_count")}건 비교`,
  };
}

function ragSignal(event: TraceResponse["events"][number] | undefined): ModelSignal | null {
  if (event === undefined) {
    return null;
  }
  const cases = parseSimilarCases(event.summary["similar_cases"]);
  const topCase = cases[0];
  return {
    source: "RAG",
    label: topCase?.label_name ?? "근거 검색",
    confidence: topCase?.similarity ?? null,
    detail: `${summaryValue(event, "document_count")}개 문서 · ${summaryValue(event, "top_title")}`,
  };
}

function fusionSignal(event: TraceResponse["events"][number] | undefined): ModelSignal | null {
  if (event === undefined) {
    return null;
  }
  return {
    source: "융합",
    label: summaryValue(event, "final_label_name"),
    confidence: numericSummaryValue(event, "confidence"),
    detail: agreementLabel(summaryValue(event, "agreement_level")),
  };
}

function ragDocumentsFromTrace(trace: TraceResponse): readonly RagDocument[] {
  const documents = findTraceEvent(trace, "rag_tool")?.summary["documents"];
  if (!Array.isArray(documents)) {
    return [];
  }
  return documents.filter(isRagDocument);
}

function isRagDocument(value: unknown): value is RagDocument {
  const record = asRecord(value);
  return (
    record !== null &&
    typeof record["document_id"] === "string" &&
    typeof record["title"] === "string" &&
    typeof record["source"] === "string" &&
    typeof record["excerpt"] === "string" &&
    typeof record["relevance"] === "number" &&
    (typeof record["source_type"] === "string" || record["source_type"] === null) &&
    asRecord(record["metadata"]) !== null
  );
}

function modelConflict(signals: readonly ModelSignal[]): boolean {
  const labels = signals
    .filter((signal) => !["없음", "근거 검색", "n/a"].includes(signal.label))
    .map((signal) => signal.label);
  return new Set(labels).size > 1;
}

function numericSummaryValue(event: TraceResponse["events"][number], key: string): number | null {
  const value = event.summary[key];
  return typeof value === "number" ? value : null;
}

function numericRecordValue(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" ? value : null;
}

function phasePosition(index: number, total: number): number {
  if (total <= 1) {
    return 0;
  }
  return Math.max(0, Math.min(100, (index / total) * 100));
}

function phaseWidth(start: number, end: number, total: number): number {
  if (total <= 1) {
    return 2;
  }
  return Math.max(2, Math.min(100, ((end - start + 1) / total) * 100));
}

function actionLabel(action: string): string {
  const labels: Record<string, string> = {
    approve: "승인",
    dispatch_field_team: "현장 출동",
    mark_false_positive: "오탐 처리",
    request_retest: "재측정 요청",
  };
  return labels[action] ?? action;
}

function operationRecordLabel(
  item: DiagnosisDetailResponse["actions"][number] | DiagnosisDetailResponse["comments"][number],
): string {
  const action = asRecord(item)?.["action"];
  return typeof action === "string" ? actionLabel(action) : "메모";
}

function timelineKindLabel(kind: string): string {
  const labels: Record<string, string> = {
    diagnosis: "진단",
    trace: "추적",
  };
  return labels[kind] ?? kind;
}

function metadataFromJson(value: unknown, current: MetadataForm): MetadataForm {
  const root = asRecord(value);
  if (root === null) {
    throw new Error("metadata json must be an object");
  }
  const metadataNode = asRecord(root["metadata"]) ?? root;
  const equipment = asRecord(metadataNode["equipment_information"]) ?? metadataNode;
  const environment = asRecord(metadataNode["environment"]) ?? metadataNode;
  return {
    equipmentName: firstString(current.equipmentName, root, metadataNode, equipment, "equipmentName", "equipment_name"),
    equipmentType: firstString(current.equipmentType, root, metadataNode, equipment, "equipmentType", "equipment_type"),
    ratedVoltage: firstString(current.ratedVoltage, root, metadataNode, equipment, "ratedVoltage", "equipment_rated_voltage"),
    ratedCurrent: firstString(current.ratedCurrent, root, metadataNode, equipment, "ratedCurrent", "equipment_rated_current"),
    sensorType: firstString(current.sensorType, root, metadataNode, environment, "sensorType", "sensor_type"),
    measurementLocation: firstString(current.measurementLocation, root, metadataNode, environment, "measurementLocation", "measurement_location"),
    operatingCondition: firstString(current.operatingCondition, root, metadataNode, environment, "operatingCondition", "operating_condition"),
    temperature: firstString(current.temperature, root, metadataNode, environment, "temperature"),
    humidity: firstString(current.humidity, root, metadataNode, environment, "humidity"),
    insulatorType: firstString(current.insulatorType, root, metadataNode, equipment, "insulatorType", "insulator_type"),
    clearanceDistance: firstString(current.clearanceDistance, root, metadataNode, environment, "clearanceDistance", "clearance_distance"),
  };
}

function firstString(fallback: string | undefined, ...sourcesAndKeys: readonly unknown[]): string {
  const keys = sourcesAndKeys.filter((item): item is string => typeof item === "string");
  const sources = sourcesAndKeys.filter((item): item is Record<string, unknown> => asRecord(item) !== null);
  for (const source of sources) {
    for (const key of keys) {
      const value = stringValue(source[key]);
      if (value !== undefined) {
        return value;
      }
    }
  }
  return fallback ?? "";
}

function stringValue(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim().length > 0) {
    return value.trim();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return undefined;
}

function evidenceSourceTitle(event: TraceResponse["events"][number] | undefined): string {
  if (event === undefined) {
    return "미실행";
  }
  const title = event.summary["model_name"] ?? event.summary["strategy"] ?? event.summary["agent"];
  return title === undefined || title === null ? "근거 수집" : String(title);
}

function summaryValue(event: TraceResponse["events"][number] | undefined, key: string): string {
  if (event === undefined) {
    return "없음";
  }
  const value = event.summary[key] ?? (key === "label_name" ? event.summary["final_label_name"] : undefined);
  if (value === undefined || value === null) {
    return "없음";
  }
  return String(value);
}

function summaryConfidence(event: TraceResponse["events"][number] | undefined): string {
  if (event === undefined) {
    return "없음";
  }
  const value = event.summary["confidence"];
  return typeof value === "number" ? `${Math.round(value * 100)}%` : summaryValue(event, "confidence");
}

function evidenceMetricRows(event: TraceResponse["events"][number] | undefined): readonly { readonly label: string; readonly value: string }[] {
  if (event?.name === "rag_tool") {
    return [
      {label: "문서 수", value: summaryValue(event, "document_count")},
      {label: "상위 근거", value: summaryValue(event, "top_title")},
      {label: "검색 질의", value: summaryValue(event, "query")},
    ];
  }
  return [
    {label: "버전", value: summaryValue(event, "model_version")},
    {label: "라벨", value: summaryValue(event, "label_name")},
    {label: "신뢰도", value: summaryConfidence(event)},
  ];
}

function evidencePayload(event: TraceResponse["events"][number] | undefined): unknown | null {
  if (event === undefined) {
    return null;
  }
  if (event.name === "fusion_engine") {
    const fusion = parseFusionSummary(event.summary);
    return fusion === null ? event.summary : compactFusionPayload(fusion);
  }
  const evidence = parseStandardEvidence(event.summary["standard_evidence"]);
  if (evidence !== null) {
    return compactStandardEvidencePayload(evidence);
  }
  const cases = compactSimilarCasesPayload(event.summary["cases"]);
  if (cases !== null) {
    return cases;
  }
  if (event.summary["evidence"] !== undefined) {
    return event.summary["evidence"];
  }
  if (event.summary["documents"] !== undefined) {
    return compactDocumentsPayload(event.summary["documents"]);
  }
  if (event.name === "metadata_context") {
    return compactMetadataPayload(event.summary);
  }
  return null;
}

function compactMetadataPayload(summary: Record<string, unknown>): unknown {
  return {
    설비명: summary["equipment_name"] ?? "없음",
    설비유형: summary["equipment_type"] ?? "없음",
    정격전압: summary["equipment_rated_voltage"] ?? "없음",
    정격전류: summary["equipment_rated_current"] ?? "없음",
    센서: summary["sensor_type"] ?? "없음",
    측정위치: summary["measurement_location"] ?? "없음",
    운전상태: summary["operating_condition"] ?? "없음",
    온도: summary["temperature"] ?? "없음",
    습도: summary["humidity"] ?? "없음",
    절연유형: summary["insulator_type"] ?? "없음",
    이격거리: summary["clearance_distance"] ?? "없음",
  };
}

function compactStandardEvidencePayload(evidence: StandardModelEvidence): unknown {
  return {
    출처: evidenceSourceLabel(evidence.source),
    모델: `${evidence.model_name}@${evidence.model_version}`,
    판정: evidence.label_name ?? "없음",
    신뢰도: formatMetric(evidence.confidence),
    불확실성: formatMetric(evidence.uncertainty),
    분포밖점수: formatMetric(evidence.ood_score),
    설명: evidence.explanation,
    주요요인: evidence.top_factors.map((factor) => ({
      항목: factorLabel(factor.name),
      값: factor.value ?? "없음",
      가중치: formatMetric(factor.weight),
      설명: factor.explanation,
    })),
  };
}

function compactFusionPayload(fusion: FusionSummaryPayload): unknown {
  return {
    최종판정: fusion.final_label_name ?? "없음",
    신뢰도: formatMetric(fusion.confidence),
    합의수준: agreementLabel(fusion.agreement_level),
    반영근거: fusion.contributing_sources.map(evidenceSourceLabel),
    판단근거: fusion.rationale,
    근거모델: fusion.evidence.map((evidence) => ({
      출처: evidenceSourceLabel(evidence.source),
      판정: evidence.label_name ?? "없음",
      신뢰도: formatMetric(evidence.confidence),
    })),
  };
}

function evidenceSourceLabel(source: string): string {
  const labels: Record<string, string> = {
    rag: "지식 검색",
    similar_case: "유사 사례",
    time_series: "시계열",
    vision: "비전",
    vlm: "VLM",
  };
  return labels[source] ?? source;
}

function agreementLabel(level: string): string {
  const labels: Record<string, string> = {
    agreement: "합의",
    conflict: "충돌",
    none: "근거 없음",
    partial_agreement: "부분 합의",
    single_source: "단일 근거",
  };
  return labels[level] ?? level;
}

function compactSimilarCasesPayload(value: unknown): unknown | null {
  const cases = parseSimilarCases(value);
  if (cases.length === 0) {
    return null;
  }
  return cases.map((item) => ({
    사례ID: item.sample_id,
    라벨: item.label_name,
    유사도: formatMetric(item.similarity),
    근거: item.reason,
  }));
}

function compactDocumentsPayload(value: unknown): unknown {
  if (!Array.isArray(value)) {
    return value;
  }
  return value.map((item) => {
    const record = asRecord(item);
    if (record === null) {
      return item;
    }
    const relevance = record["relevance"];
    return {
      유형: sourceTypeLabel(stringValue(record["source_type"]) ?? stringValue(asRecord(record["metadata"])?.["source_type"])),
      문서: record["title"] ?? "없음",
      출처: record["source"] ?? "없음",
      관련도: typeof relevance === "number" ? formatMetric(relevance) : "없음",
      요약: record["excerpt"] ?? "없음",
    };
  });
}

function sourceTypeLabel(sourceType: string | undefined): string {
  const labels: Record<string, string> = {
    dataset_case: "데이터셋 사례",
    rulebook: "규칙서",
    sop: "SOP",
  };
  return sourceType === undefined ? "없음" : labels[sourceType] ?? sourceType;
}

function ragSourceTypes(status: RagStatusResponse | null): readonly string[] {
  if (status !== null) {
    const sourceTypes = [...status.source_types, ...Object.keys(status.source_counts)];
    if (sourceTypes.length > 0) {
      return Array.from(new Set(sourceTypes));
    }
  }
  return ["rulebook", "dataset_case", "sop"];
}

function sourceTypeCount(sourceType: string, status: RagStatusResponse | null): number {
  if (status === null) {
    return 0;
  }
  if (sourceType === "all") {
    return status.document_count;
  }
  return status.source_counts[sourceType]?.documents ?? 0;
}

function boundedRagTopK(value: number): number {
  if (!Number.isFinite(value)) {
    return RAG_TOP_K_MIN;
  }
  return Math.min(RAG_TOP_K_MAX, Math.max(RAG_TOP_K_MIN, Math.trunc(value)));
}

function parseDatasetLimit(value: string): number | null {
  if (value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.min(RAG_DATASET_LIMIT_MAX, Math.max(RAG_DATASET_LIMIT_MIN, Math.trunc(parsed)));
}

function factorLabel(name: string): string {
  const labels: Record<string, string> = {
    abs_p99: "상위 진폭 분위값",
    band_like_noise_score: "대역형 노이즈 점수",
    ood_score: "분포 밖 점수",
    phase_localization_score: "위상 국부화 점수",
    pulse_rate: "펄스 반복률",
    reason: "판단 사유",
    recommended_action: "권고 조치",
    spectral_energy: "주파수 에너지",
  };
  return labels[name] ?? name;
}

function parseFusionSummary(value: unknown): FusionSummaryPayload | null {
  const record = asRecord(value);
  if (record === null) {
    return null;
  }
  const evidence = parseStandardEvidenceArray(record["evidence"]);
  if (
    typeof record["strategy"] !== "string" ||
    !(typeof record["final_label_id"] === "number" || record["final_label_id"] === null) ||
    !(typeof record["final_label_name"] === "string" || record["final_label_name"] === null) ||
    !(typeof record["confidence"] === "number" || record["confidence"] === null) ||
    typeof record["agreement_level"] !== "string" ||
    !isStringArray(record["contributing_sources"]) ||
    typeof record["rationale"] !== "string" ||
    evidence === null
  ) {
    return null;
  }
  return {
    strategy: record["strategy"],
    final_label_id: record["final_label_id"],
    final_label_name: record["final_label_name"],
    confidence: record["confidence"],
    agreement_level: record["agreement_level"],
    contributing_sources: record["contributing_sources"],
    rationale: record["rationale"],
    evidence,
  };
}

function parseStandardEvidenceArray(value: unknown): readonly StandardModelEvidence[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const evidence = value.map(parseStandardEvidence);
  if (evidence.some((item) => item === null)) {
    return null;
  }
  return evidence as readonly StandardModelEvidence[];
}

function parseStandardEvidence(value: unknown): StandardModelEvidence | null {
  const record = asRecord(value);
  if (record === null || !isEvidenceSource(record["source"])) {
    return null;
  }
  const factors = parseEvidenceFactors(record["top_factors"]);
  if (
    typeof record["model_name"] !== "string" ||
    typeof record["model_version"] !== "string" ||
    !(typeof record["label_id"] === "number" || record["label_id"] === null) ||
    !(typeof record["label_name"] === "string" || record["label_name"] === null) ||
    !(typeof record["confidence"] === "number" || record["confidence"] === null) ||
    !(typeof record["uncertainty"] === "number" || record["uncertainty"] === null) ||
    !(typeof record["ood_score"] === "number" || record["ood_score"] === null) ||
    factors === null ||
    typeof record["explanation"] !== "string"
  ) {
    return null;
  }
  return {
    source: record["source"],
    model_name: record["model_name"],
    model_version: record["model_version"],
    label_id: record["label_id"],
    label_name: record["label_name"],
    confidence: record["confidence"],
    uncertainty: record["uncertainty"],
    ood_score: record["ood_score"],
    top_factors: factors,
    explanation: record["explanation"],
  };
}

function parseEvidenceFactors(value: unknown): readonly EvidenceFactor[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const factors = value.filter(isEvidenceFactor);
  return factors.length === value.length ? factors : null;
}

function isEvidenceFactor(value: unknown): value is EvidenceFactor {
  const record = asRecord(value);
  return (
    record !== null &&
    typeof record["name"] === "string" &&
    (typeof record["value"] === "string" || typeof record["value"] === "number" || record["value"] === null) &&
    typeof record["weight"] === "number" &&
    typeof record["explanation"] === "string"
  );
}

function isEvidenceSource(value: unknown): value is StandardModelEvidence["source"] {
  return value === "time_series" || value === "vision" || value === "vlm" || value === "rag" || value === "similar_case";
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function parseSimilarCases(value: unknown): readonly SimilarCase[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(isSimilarCase);
}

function isSimilarCase(value: unknown): value is SimilarCase {
  const record = asRecord(value);
  if (record === null) {
    return false;
  }
  return (
    typeof record["sample_id"] === "string" &&
    typeof record["label_id"] === "number" &&
    typeof record["label_name"] === "string" &&
    typeof record["equipment_name"] === "string" &&
    typeof record["insulator_type"] === "string" &&
    typeof record["sensor_type"] === "string" &&
    typeof record["clearance_distance"] === "string" &&
    typeof record["similarity"] === "number" &&
    typeof record["reason"] === "string" &&
    typeof record["image_url"] === "string" &&
    asRecord(record["metadata"]) !== null
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function similarityWidth(value: number): number {
  return Math.max(2, Math.min(100, Math.round(value * 100)));
}

function metadataValue(item: SimilarCase, key: string): string {
  const value = item.metadata[key];
  if (value === undefined || value === null || value === "") {
    return "없음";
  }
  return String(value);
}

async function loadDashboard(): Promise<{
  backendStatus: BackendStatus;
  history: readonly DiagnosisListItem[];
  modelRuntime: ModelRuntimeStatus;
  reviewQueue: readonly DiagnosisListItem[];
}> {
  const [health, history, modelRuntime, reviewQueue] = await Promise.all([
    fetchHealth(),
    fetchDiagnosisHistory(),
    fetchModelRuntimeStatus(),
    fetchReviewQueue(),
  ]);
  return {
    backendStatus: health.status === "ok" ? "online" : "offline",
    history: history.items,
    modelRuntime,
    reviewQueue: reviewQueue.items,
  };
}

function backendStatusLabel(status: BackendStatus): string {
  if (status === "online") {
    return "온라인";
  }
  if (status === "offline") {
    return "오프라인";
  }
  return "확인 중";
}

function backendTone(status: BackendStatus): string {
  if (status === "online") {
    return "green";
  }
  if (status === "offline") {
    return "red";
  }
  return "blue";
}

function statusTone(status: DiagnosisStatus | undefined): string {
  if (status === "completed") {
    return "green";
  }
  if (status === "needs_review") {
    return "amber";
  }
  if (status === "rejected") {
    return "red";
  }
  return "violet";
}

function modelRuntimeAction(status: ModelRuntimeStatus | null): string {
  if (status === null) {
    return "확인 중";
  }
  if (status.time_series_ready && status.vision_ready && status.vlm_ready) {
    return status.adapter_mode;
  }
  return "확인 필요";
}

function formatConfidence(confidence: number | null): string {
  return confidence === null ? "없음" : `${Math.round(confidence * 100)}%`;
}

function formatMetric(metric: number | null): string {
  return metric === null ? "없음" : `${Math.round(metric * 100)}%`;
}

function formatNumeric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "없음";
  }
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, {maximumFractionDigits: 0});
  }
  if (Math.abs(value) >= 10) {
    return value.toLocaleString(undefined, {maximumFractionDigits: 2});
  }
  return value.toLocaleString(undefined, {maximumFractionDigits: 4});
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function traceEventKey(kind: string, name: string, index: number): string {
  return `${index}:${kind}:${name}`;
}

function validateImageFile(file: File | null): { readonly label: string; readonly message: string; readonly ok: boolean } {
  if (file === null) {
    return {label: "대기", message: "PRPD 이미지가 선택되지 않았습니다.", ok: false};
  }
  const pngLike = file.type === "image/png" || file.name.toLowerCase().endsWith(".png");
  return {
    label: pngLike ? "준비" : "확인",
    message: pngLike ? "PNG 이미지가 선택되었습니다." : "PRPD 경로에는 PNG 파일을 제출해야 합니다.",
    ok: pngLike,
  };
}

function validateCsvFile(file: File | null): { readonly label: string; readonly message: string; readonly ok: boolean } {
  if (file === null) {
    return {label: "대기", message: "시계열 CSV가 선택되지 않았습니다.", ok: false};
  }
  const csvLike = file.type === "text/csv" || file.name.toLowerCase().endsWith(".csv");
  return {
    label: csvLike ? "준비" : "확인",
    message: csvLike ? "CSV 파일이 선택되었습니다. 제출 시 서버가 신호 형식을 확인합니다." : ".csv 파일을 제출해야 합니다.",
    ok: csvLike,
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kib = bytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(1)} KiB`;
  }
  return `${(kib / 1024).toFixed(1)} MiB`;
}

function downloadJson(filename: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
