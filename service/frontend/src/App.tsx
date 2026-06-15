import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Cpu,
  Database,
  Download,
  X,
  FileSearch,
  FileCheck2,
  FileImage,
  FileJson,
  FileText,
  Moon,
  Gauge,
  Image as ImageIcon,
  Loader2,
  MessageSquare,
  Printer,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  ShieldCheck,
  Sun,
  UploadCloud,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  apiAssetUrl,
  askRagChat,
  fetchDatasetCaseDetail,
  fetchRagDocumentDetail,
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
  RagAppliedFilter,
  RagDocument,
  RagDocumentListItem,
  RagChatMessage,
  RagDocumentDetailResponse,
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
type RagChatStatus = "idle" | "asking" | "failed";
type RagDocumentDetailStatus = "idle" | "loading" | "failed";
type RuntimeRefreshStatus = "idle" | "loading" | "failed";
type SimilarCaseDetailStatus = "idle" | "loading" | "failed";
type ThemeMode = "light" | "dark";

const THEME_STORAGE_KEY = "pd-theme-mode";

const RAG_TOP_K_MIN = 1;
const RAG_TOP_K_MAX = 20;
const RAG_DATASET_LIMIT_MIN = 1;
const RAG_DATASET_LIMIT_MAX = 50_000;
const DEFAULT_SIMILAR_CASE_LIMIT = 3;
const REPORT_RAG_DOCUMENT_LIMIT = 5;
const REPORT_RAG_FACT_LIMIT = 6;

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

type ReportEvidenceLink = {
  readonly kind: "document" | "case";
  readonly reason: string;
  readonly score: number | null;
  readonly title: string;
};

type ContributionItem = {
  readonly details: readonly string[];
  readonly label: string;
  readonly score: number | null;
  readonly summary: string;
};

type SimilarityComponentItem = {
  readonly label: string;
  readonly score: number;
};

type SimilarCaseRankSummary = {
  readonly detail: string;
  readonly title: string;
};

type CaseComparisonRow = {
  readonly currentValue: string;
  readonly label: string;
  readonly match: "same" | "different" | "unknown";
  readonly similarValue: string;
};

type CaseComparisonPoint = {
  readonly detail: string;
  readonly title: string;
  readonly tone: "same" | "different" | "neutral";
};

type LlmRagFlowStep = {
  readonly detail: string;
  readonly label: string;
  readonly state: "ready" | "warn" | "neutral";
  readonly title: string;
};

type RagFact = {
  readonly label: string;
  readonly value: string;
};

type ReportRagDocument = {
  readonly document: RagDocument;
  readonly facts: readonly RagFact[];
  readonly hiddenFactCount: number;
  readonly rank: number;
  readonly retrievalLabel: string;
  readonly scoreTone: string;
  readonly sourceLabel: string;
  readonly summary: string;
};

type RagEvidenceSummary = {
  readonly documentCount: number;
  readonly retrievalText: string;
  readonly shownCount: number;
  readonly sourceText: string;
  readonly topScore: number | null;
};

type RagChatTurn = RagChatMessage & {
  readonly answerMode?: string | null;
  readonly documents?: readonly RagDocument[];
  readonly error?: string | null;
  readonly model?: string | null;
};

type RagChatContentBlock =
  | {readonly kind: "heading"; readonly text: string}
  | {readonly kind: "paragraph"; readonly text: string}
  | {readonly kind: "list"; readonly items: string[]};

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

const DEFAULT_INTAKE_FILES = {
  csv: {
    mimeType: "text/csv",
    name: "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.csv",
    url: "/demo-intake/default-timeseries.csv",
  },
  image: {
    mimeType: "image/png",
    name: "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png",
    url: "/demo-intake/default-prpd.png",
  },
  metadata: {
    mimeType: "application/json",
    name: "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.json",
    url: "/demo-intake/default-metadata.json",
  },
} as const;

const EMPTY_METADATA: MetadataForm = {
  equipmentName: "",
  equipmentType: "",
  ratedVoltage: "",
  ratedCurrent: "",
  sensorType: "",
  measurementLocation: "",
  operatingCondition: "",
  temperature: "",
  humidity: "",
  insulatorType: "",
  clearanceDistance: "",
};

const dashboardViewIds = [
  "overview",
  "intake",
  "verdict",
  "evidence",
  "case-search",
  "detail",
  "report",
  "rag-admin",
  "rag-chat",
  "history",
  "queue",
  "trace",
] as const;

type DashboardViewId = typeof dashboardViewIds[number];

type NavItem = {
  readonly id: DashboardViewId;
  readonly icon: ReactNode;
  readonly label: string;
};

function viewFromHash(hash: string): DashboardViewId {
  const candidate = hash.replace("#", "");
  if (candidate === "model-runtime") {
    return "overview";
  }
  return dashboardViewIds.includes(candidate as DashboardViewId) ? candidate as DashboardViewId : "overview";
}

async function loadDefaultIntakeFiles(): Promise<{
  readonly csv: File;
  readonly image: File;
  readonly metadata: File;
}> {
  const [image, csv, metadata] = await Promise.all([
    loadPublicFile(DEFAULT_INTAKE_FILES.image),
    loadPublicFile(DEFAULT_INTAKE_FILES.csv),
    loadPublicFile(DEFAULT_INTAKE_FILES.metadata),
  ]);
  return {image, csv, metadata};
}

async function loadPublicFile(asset: {readonly mimeType: string; readonly name: string; readonly url: string}): Promise<File> {
  const response = await fetch(asset.url);
  if (!response.ok) {
    throw new Error(`default intake file request failed: ${response.status}`);
  }
  const blob = await response.blob();
  return new File([blob], asset.name, {type: asset.mimeType});
}

function getInitialThemeMode(): ThemeMode {
  if (typeof window === "undefined") {
    return "light";
  }
  const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (saved === "dark" || saved === "light") {
    return saved;
  }
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export function App() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(getInitialThemeMode);
  const [activeView, setActiveView] = useState<DashboardViewId>(() => viewFromHash(window.location.hash));
  const [image, setImage] = useState<File | null>(null);
  const [csv, setCsv] = useState<File | null>(null);
  const [metadataJsonStatus, setMetadataJsonStatus] = useState("메타데이터 JSON 업로드 대기");
  const [metadata, setMetadata] = useState<MetadataForm>(DEFAULT_METADATA);
  const [uploadResetKey, setUploadResetKey] = useState(0);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [detail, setDetail] = useState<DiagnosisDetailResponse | null>(null);
  const [history, setHistory] = useState<readonly DiagnosisListItem[]>([]);
  const [reviewQueue, setReviewQueue] = useState<readonly DiagnosisListItem[]>([]);
  const [selectedHistory, setSelectedHistory] = useState<DiagnosisListItem | null>(null);
  const [expandedTraceKey, setExpandedTraceKey] = useState<string | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [requestStatus, setRequestStatus] = useState<RequestStatus>("idle");
  const [requestError, setRequestError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [modelRuntime, setModelRuntime] = useState<ModelRuntimeStatus | null>(null);
  const [ragStatus, setRagStatus] = useState<RagStatusResponse | null>(null);
  const [selectedSimilarCase, setSelectedSimilarCase] = useState<SimilarCase | null>(null);
  const [similarCaseDetailStatus, setSimilarCaseDetailStatus] = useState<SimilarCaseDetailStatus>("idle");
  const [currentSimilarCaseLimit, setCurrentSimilarCaseLimit] = useState(DEFAULT_SIMILAR_CASE_LIMIT);
  const intakeTouchedRef = useRef(false);
  const route = useMemo(
    () => selectInputRoute(buildInputPresence({ hasImage: image !== null, hasTimeseries: csv !== null, metadata })),
    [image, csv, metadata],
  );
  const activeRoute = result?.route ?? selectedHistory?.route ?? route;
  const verdictStatus = result?.status ?? selectedHistory?.status;
  const currentSimilarCases = useMemo(() => similarCasesFromTrace(trace), [trace]);
  const navItems: readonly NavItem[] = [
    {id: "overview", icon: <Gauge size={18} />, label: "현황"},
    {id: "intake", icon: <Database size={18} />, label: "진단 접수"},
    {id: "verdict", icon: <ShieldCheck size={18} />, label: "판정"},
    {id: "evidence", icon: <BarChart3 size={18} />, label: "근거"},
    {id: "case-search", icon: <ImageIcon size={18} />, label: "과거 사례 비교"},
    {id: "detail", icon: <FileCheck2 size={18} />, label: "진단 기록"},
    {id: "report", icon: <FileSearch size={18} />, label: "리포트"},
    {id: "rag-admin", icon: <BookOpen size={18} />, label: "RAG 관리"},
    {id: "rag-chat", icon: <MessageSquare size={18} />, label: "RAG 챗"},
    {id: "history", icon: <Database size={18} />, label: "이력"},
    {id: "queue", icon: <AlertTriangle size={18} />, label: "검토 큐"},
    {id: "trace", icon: <Cpu size={18} />, label: "추적"},
  ];
  const activeNavLabel = navItems.find((item) => item.id === activeView)?.label ?? "현황";

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    let active = true;
    void loadDashboard()
      .then((dashboard) => {
        if (active) {
          setBackendStatus(dashboard.backendStatus);
          setHistory(dashboard.history);
          setReviewQueue(dashboard.reviewQueue);
          setModelRuntime(dashboard.modelRuntime);
          setRagStatus(dashboard.ragStatus);
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
    const syncViewFromHash = () => {
      setActiveView(viewFromHash(window.location.hash));
      window.scrollTo({top: 0});
    };
    window.addEventListener("hashchange", syncViewFromHash);
    return () => window.removeEventListener("hashchange", syncViewFromHash);
  }, []);

  useEffect(() => {
    let active = true;
    void loadDefaultIntakeFiles()
      .then((files) => {
        if (!active || intakeTouchedRef.current) {
          return;
        }
        setImage(files.image);
        setCsv(files.csv);
        void handleMetadataJson(files.metadata);
      })
      .catch(() => {
        if (active && !intakeTouchedRef.current) {
          setMetadataJsonStatus("기본 데모 입력 불러오기 실패");
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
    setRequestError(null);
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
      selectView("verdict");
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "진단 요청을 처리하지 못했습니다.");
      setRequestStatus("failed");
    }
  }

  function handleImageFile(file: File | null): void {
    intakeTouchedRef.current = true;
    setImage(file);
  }

  function handleCsvFile(file: File | null): void {
    intakeTouchedRef.current = true;
    setCsv(file);
  }

  function handleMetadataFile(file: File | null): void {
    intakeTouchedRef.current = true;
    void handleMetadataJson(file);
  }

  function resetIntakeForm(): void {
    intakeTouchedRef.current = true;
    setImage(null);
    setCsv(null);
    setMetadata(EMPTY_METADATA);
    setMetadataJsonStatus("메타데이터 JSON 업로드 대기");
    setRequestError(null);
    setRequestStatus("idle");
    setResult(null);
    setTrace(null);
    setDetail(null);
    setSelectedHistory(null);
    setExpandedTraceKey(null);
    setUploadResetKey((current) => current + 1);
  }

  function handleThemeModeChange(mode: ThemeMode): void {
    setThemeMode(mode);
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

  async function refreshModelRuntime(): Promise<void> {
    setModelRuntime(await fetchModelRuntimeStatus());
  }

  async function openDiagnosis(item: DiagnosisListItem): Promise<void> {
    setSelectedHistory(item);
    setResult(null);
    setExpandedTraceKey(null);
    const nextDetail = await fetchDiagnosisDetail(item.diagnosis_id);
    setTrace(nextDetail.trace);
    setDetail(nextDetail);
    selectView("detail");
  }

  async function downloadReport(): Promise<void> {
    const diagnosisId = detail?.diagnosis.diagnosis_id ?? result?.diagnosis_id;
    if (diagnosisId === undefined) {
      return;
    }
    const report = await fetchDiagnosisReport(diagnosisId);
    downloadJson(`${diagnosisId}-diagnosis-report.json`, report);
  }

  async function openSimilarCase(item: SimilarCase): Promise<void> {
    setSelectedSimilarCase(item);
    setSimilarCaseDetailStatus("loading");
    try {
      const detailCase = await fetchDatasetCaseDetail(item.sample_id);
      setSelectedSimilarCase({
        ...detailCase,
        reason: item.reason,
        similarity: item.similarity,
        metadata: {
          ...detailCase.metadata,
          ...item.metadata,
        },
      });
      setSimilarCaseDetailStatus("idle");
    } catch {
      setSimilarCaseDetailStatus("failed");
    }
  }

  function selectView(viewId: DashboardViewId): void {
    setActiveView(viewId);
    const nextHash = `#${viewId}`;
    if (window.location.hash !== nextHash) {
      window.history.pushState(null, "", nextHash);
    }
    window.scrollTo({top: 0});
  }

  function renderActiveView(): ReactNode {
    if (activeView === "overview") {
      return (
        <OverviewPanel
          activeRoute={routeLabel[activeRoute]}
          backendStatus={backendStatus}
          currentSimilarCaseCount={currentSimilarCases.length}
          modelRuntime={modelRuntime}
          ragStatus={ragStatus}
          reviewQueue={reviewQueue}
        />
      );
    }

    if (activeView === "intake") {
      return (
        <Panel id="intake" title="진단 접수" action={routeLabel[route]}>
          <form className="intakeForm" onSubmit={(event) => void handleSubmit(event)}>
            <div className="uploadGrid">
              <UploadField
                accept=".png,image/png"
                description={fileDisplayText(image, "PRPD 이미지")}
                icon={<FileImage size={20} />}
                key={`image-${uploadResetKey}`}
                label="PRPD 이미지"
                onChange={handleImageFile}
              />
              <UploadField
                accept=".csv,text/csv,application/vnd.ms-excel"
                description={fileDisplayText(csv, "시계열 CSV")}
                icon={<FileText size={20} />}
                key={`csv-${uploadResetKey}`}
                label="시계열 CSV"
                onChange={handleCsvFile}
              />
              <UploadField
                accept=".json,application/json"
                description={metadataJsonStatus}
                icon={<FileJson size={20} />}
                key={`metadata-${uploadResetKey}`}
                label="메타데이터 JSON"
                onChange={handleMetadataFile}
              />
            </div>

            <InputInspector csv={csv} image={image} imagePreviewUrl={imagePreviewUrl} />
            {requestError === null ? null : (
              <div className="formAlert" role="alert">
                <AlertTriangle size={16} />
                <span>{requestError}</span>
              </div>
            )}

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
      );
    }

    if (activeView === "verdict") {
      return (
        <Panel id="verdict" title="진단 판정" action={verdictStatus === undefined ? "대기" : statusLabel[verdictStatus]}>
          <VerdictPanel historyItem={selectedHistory} result={result} requestError={requestError} requestStatus={requestStatus} />
        </Panel>
      );
    }

    if (activeView === "evidence") {
      return (
        <Panel id="evidence" title="진단 근거" action={trace?.trace_id ?? "대기"}>
          <EvidencePanel trace={trace} onOpenCase={(item) => void openSimilarCase(item)} />
        </Panel>
      );
    }

    if (activeView === "case-search") {
      return (
        <Panel id="case-search" title="과거 사례 비교" action={trace === null ? "대기" : `${currentSimilarCases.length}건`}>
          <CurrentSimilarCasesPanel
            cases={currentSimilarCases}
            hasTrace={trace !== null}
            limit={boundedSimilarCaseLimit(currentSimilarCaseLimit, currentSimilarCases.length)}
            requestStatus={requestStatus}
            onLimitChange={setCurrentSimilarCaseLimit}
            onOpenCase={(item) => void openSimilarCase(item)}
          />
        </Panel>
      );
    }

    if (activeView === "detail") {
      return (
        <Panel id="detail" title="진단 기록" action={detail?.diagnosis.diagnosis_id ?? "선택 없음"}>
          <DetailPanel detail={detail} onDownload={() => void downloadReport()} onOpenCase={(item) => void openSimilarCase(item)} />
        </Panel>
      );
    }

    if (activeView === "report") {
      return (
        <Panel id="report" title="운영 리포트" action={detail?.diagnosis.status === undefined ? "대기" : statusLabel[detail.diagnosis.status]}>
          <ReportPanel detail={detail} onDownload={() => void downloadReport()} onOpenCase={(item) => void openSimilarCase(item)} />
        </Panel>
      );
    }

    if (activeView === "rag-admin") {
      return (
        <Panel id="rag-admin" title="RAG / LLM 리포터 관리" action="검색·리포터·재색인">
          <RagAdminPanel runtimeStatus={modelRuntime} onRefreshRuntime={refreshModelRuntime} />
        </Panel>
      );
    }

    if (activeView === "rag-chat") {
      return (
        <Panel id="rag-chat" title="RAG 챗" action={modelRuntime?.llm_rag_ready ? "OpenRouter 연결" : "키 설정 필요"}>
          <RagChatPanel runtimeStatus={modelRuntime} />
        </Panel>
      );
    }

    if (activeView === "history") {
      return (
        <Panel id="history" title="진단 이력" action={`${history.length}건`}>
          <DiagnosisTable items={history} onOpen={(item) => void openDiagnosis(item)} />
        </Panel>
      );
    }

    if (activeView === "queue") {
      return (
        <Panel id="queue" title="검토 대기" action={`${reviewQueue.length}건`}>
          <ReviewQueue items={reviewQueue} onOpen={(item) => void openDiagnosis(item)} />
        </Panel>
      );
    }

    if (activeView === "trace") {
      return (
        <section id="trace" className="panel">
          <div className="panelHeader">
            <h3>처리 추적</h3>
            <span>{trace === null ? "추적 없음" : `${trace.trace_id} · ${trace.events.length}개 이벤트`}</span>
          </div>
          <TraceLog expandedKey={expandedTraceKey} onToggle={setExpandedTraceKey} trace={trace} />
        </section>
      );
    }

    return null;
  }

  return (
    <>
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
          <div className="plantBadgeHeader">
            <div>
              <span>공정 현장</span>
              <strong>설비 진단 콘솔</strong>
            </div>
              <div className="themeChoice" role="group" aria-label="테마 선택">
                <button
                  aria-pressed={themeMode === "light"}
                  aria-label="라이트 모드"
                  title="라이트 모드"
                  className={themeMode === "light" ? "themeChoiceButton active" : "themeChoiceButton"}
                  onClick={() => handleThemeModeChange("light")}
                  type="button"
                >
                  <Sun size={14} />
                </button>
                <button
                  aria-pressed={themeMode === "dark"}
                  aria-label="다크 모드"
                  title="다크 모드"
                  className={themeMode === "dark" ? "themeChoiceButton active" : "themeChoiceButton"}
                  onClick={() => handleThemeModeChange("dark")}
                  type="button"
                >
                  <Moon size={14} />
                </button>
              </div>
            </div>
          </div>
        <nav className="navList">
          {navItems.map((item) => (
            <a
              aria-current={activeView === item.id ? "page" : undefined}
              className={activeView === item.id ? "active" : ""}
              href={`#${item.id}`}
              key={item.id}
              onClick={(event) => {
                event.preventDefault();
                selectView(item.id);
              }}
            >
              {item.icon}
              {item.label}
            </a>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">설비 관측 콘솔</p>
            <h2>{activeNavLabel}</h2>
          </div>
          <div className="topbarActions">
            {activeView === "intake" ? (
              <button className="iconButton" type="button" onClick={resetIntakeForm}>
                <RotateCcw size={18} />
                초기화
              </button>
            ) : null}
            <button className="iconButton" type="button" onClick={() => window.location.reload()}>
              <RefreshCw size={18} />
              새로고침
            </button>
          </div>
        </header>

        <div className="viewFrame" key={activeView}>
          {renderActiveView()}
        </div>
      </section>
    </main>
    <SimilarCaseDetailModal
      item={selectedSimilarCase}
      currentArtifacts={trace === null ? null : inputArtifactsFromTrace(trace)}
      trace={trace}
      status={similarCaseDetailStatus}
      onClose={() => {
        setSelectedSimilarCase(null);
        setSimilarCaseDetailStatus("idle");
      }}
    />
    </>
  );
}

function OverviewPanel(props: {
  readonly activeRoute: string;
  readonly backendStatus: BackendStatus;
  readonly currentSimilarCaseCount: number;
  readonly modelRuntime: ModelRuntimeStatus | null;
  readonly ragStatus: RagStatusResponse | null;
  readonly reviewQueue: readonly DiagnosisListItem[];
}) {
  return (
    <section id="overview" className="overviewView">
      <div className="kpiGrid viewPanel">
        <MetricCard icon={<Database size={20} />} label="서버" value={backendStatusLabel(props.backendStatus)} tone={backendTone(props.backendStatus)} />
        <MetricCard icon={<Gauge size={20} />} label="진단 경로" value={props.activeRoute} tone="blue" />
        <MetricCard icon={<AlertTriangle size={20} />} label="검토 큐" value={props.reviewQueue.length.toString()} tone={props.reviewQueue.length > 0 ? "red" : "green"} />
        <MetricCard icon={<BookOpen size={20} />} label="RAG 문서" value={props.ragStatus === null ? "확인 중" : props.ragStatus.document_count.toString()} tone={props.ragStatus?.ready ? "green" : "amber"} />
        <MetricCard icon={<Cpu size={20} />} label="현재 유사 사례" value={props.currentSimilarCaseCount.toString()} tone="violet" />
      </div>

      <div className="overviewGrid">
        <Panel id="overview-runtime" title="모델 런타임" action={modelRuntimeAction(props.modelRuntime)}>
          <ModelRuntimePanel status={props.modelRuntime} />
        </Panel>
      </div>
    </section>
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
  return (
    <label
      className="uploadField"
      onDragOver={(event) => {
        event.preventDefault();
      }}
      onDrop={(event) => {
        event.preventDefault();
        props.onChange(event.dataTransfer.files.item(0));
      }}
    >
      <div className="uploadIcon">{props.icon}</div>
      <div>
        <strong>{props.label}</strong>
        <span>{props.description}</span>
      </div>
      <span className="uploadButton">
        <UploadCloud size={16} />
        선택
      </span>
      <input
        aria-label={`${props.label} 파일 선택`}
        accept={props.accept}
        className="fileInput"
        type="file"
        onClick={(event) => {
          event.currentTarget.value = "";
        }}
        onChange={(event) => props.onChange(event.currentTarget.files?.item(0) ?? null)}
      />
    </label>
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
  readonly requestError: string | null;
  readonly result: DiagnosisResponse | null;
  readonly requestStatus: RequestStatus;
}) {
  if (props.requestStatus === "failed") {
    return (
      <div className="emptyState error">
        <AlertTriangle size={28} />
        <strong>요청 실패</strong>
        <span>{props.requestError ?? "서버 상태 또는 업로드 파일을 확인해 주세요."}</span>
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
  readonly onOpenCase: (item: SimilarCase) => void;
}) {
  const [similarCaseLimit, setSimilarCaseLimit] = useState(DEFAULT_SIMILAR_CASE_LIMIT);
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
  const activeSimilarCaseLimit = boundedSimilarCaseLimit(similarCaseLimit, referenceCases.length);
  const visibleReferenceCases = referenceCases.slice(0, activeSimilarCaseLimit);
  const timeline = props.detail.timeline;
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
        <SimilarCaseBoard
          cases={visibleReferenceCases}
          headerControl={(
            <SimilarCaseLimitControl
              max={referenceCases.length}
              value={activeSimilarCaseLimit}
              onChange={setSimilarCaseLimit}
            />
          )}
          subtitle={`상위 ${activeSimilarCaseLimit}건 표시 / 전체 ${referenceCases.length}건`}
          title="리포트 참조 사례"
          onOpenCase={props.onOpenCase}
        />
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
  readonly onOpenCase: (item: SimilarCase) => void;
}) {
  if (props.detail === null) {
    return (
      <div className="emptyState compact">
        <FileSearch size={24} />
        <strong>리포트 대기</strong>
        <span>진단 기록이 선택되면 관리자 검토용 리포트가 구성됩니다.</span>
      </div>
    );
  }

  const signals = modelSignalsFromTrace(props.detail.trace);
  const ragDocuments = ragDocumentsFromTrace(props.detail.trace);
  const reportRagDocuments = rankedReportRagDocuments(ragDocuments);
  const ragEvidenceSummary = reportRagEvidenceSummary(ragDocuments, reportRagDocuments.length);
  const referenceCases = similarCasesFromTrace(props.detail.trace);
  const fusion = parseFusionSummary(findTraceEvent(props.detail.trace, "fusion_engine")?.summary);
  const finalLabel = fusion?.final_label_name ?? props.detail.diagnosis.diagnosis ?? "판정 보류";
  const contributionItems = reportContributionItems(ragDocuments, referenceCases, fusion);
  return (
    <div className="reportCanvas">
      <article className="reportPaper">
        <section className="reportHero">
          <div className="reportTitleBlock">
            <p className="eyebrow">관리자 검토 리포트</p>
            <div className="reportVerdictLine">
              <h4>최종 판정: {finalLabel}</h4>
              <span className={`status ${props.detail.diagnosis.status}`}>{statusLabel[props.detail.diagnosis.status]}</span>
            </div>
            <p className="reportLead">{props.detail.diagnosis.reason}</p>
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

        <dl className="reportMetaStrip">
          <div><dt>진단 ID</dt><dd>{props.detail.diagnosis.diagnosis_id}</dd></div>
          <div><dt>경로</dt><dd>{routeLabel[props.detail.diagnosis.route]}</dd></div>
          <div><dt>상태</dt><dd>{statusLabel[props.detail.diagnosis.status]}</dd></div>
          <div><dt>위험도</dt><dd>{props.detail.diagnosis.risk_level ?? "없음"}</dd></div>
          <div><dt>신뢰도</dt><dd>{formatConfidence(props.detail.diagnosis.confidence)}</dd></div>
          <div><dt>Trace</dt><dd>{props.detail.diagnosis.trace_id}</dd></div>
        </dl>

        <section className="reportSection">
          <div className="reportSectionHeader">
            <span>01</span>
            <div>
              <h5>판정 근거 요약</h5>
              <p>모델 출력, 검색 근거, 융합 판단을 최종 판정 기준으로 정리했습니다.</p>
            </div>
          </div>
          <div className="reportSignalGrid">
            {signals.map((signal) => (
              <article className="reportSignalCard" key={signal.source}>
                <div className="reportSignalHeader">
                  <span>{signal.source}</span>
                  <b>{formatConfidence(signal.confidence)}</b>
                </div>
                <strong>{signal.label}</strong>
                <p title={signal.detail}>{signal.detail}</p>
                <EvidenceLinkList items={evidenceLinksForSignal(signal, finalLabel, ragDocuments, referenceCases)} />
              </article>
            ))}
          </div>
          {fusion !== null ? <p className="reportNote"><strong>{agreementLabel(fusion.agreement_level)}</strong>{fusion.rationale}</p> : null}
        </section>

        <section className="reportSection">
          <div className="reportSectionHeader">
            <span>02</span>
            <div>
              <h5>최종 판정 기여도</h5>
              <p>문서, 유사 사례, 융합 판단이 결론에 기여한 정도를 비교합니다.</p>
            </div>
          </div>
          <div className="contributionGrid">
            {contributionItems.map((item) => (
              <article className="contributionCard" key={item.label}>
                <div className="contributionHeader">
                  <span>{item.label}</span>
                  <strong>{formatMetric(item.score)}</strong>
                </div>
                <div className="contributionBar" aria-label={`${item.label} ${formatMetric(item.score)}`}>
                  <span style={{width: `${metricWidth(item.score)}%`}} />
                </div>
                <p>{item.summary}</p>
                <ul>
                  {item.details.map((detail) => (
                    <li key={detail}>{detail}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>

        <section className="reportSection">
          <div className="reportSectionHeader">
            <span>03</span>
            <div>
              <h5>RAG 근거 문서</h5>
              <p>리포터 입력에 포함된 지식 문서를 출처, 검색 방식, 관련도, 핵심 필드로 나누어 표시합니다.</p>
            </div>
          </div>
          {ragDocuments.length > 0 ? (
            <div className="ragEvidenceOverview" aria-label="RAG 근거 요약">
              <div>
                <span>표시 문서</span>
                <strong>{ragEvidenceSummary.shownCount}/{ragEvidenceSummary.documentCount}건</strong>
              </div>
              <div>
                <span>최고 관련도</span>
                <strong>{formatMetric(ragEvidenceSummary.topScore)}</strong>
              </div>
              <div>
                <span>출처 구성</span>
                <strong>{ragEvidenceSummary.sourceText}</strong>
              </div>
              <div>
                <span>검색 방식</span>
                <strong>{ragEvidenceSummary.retrievalText}</strong>
              </div>
            </div>
          ) : null}
          <div className="reportDocumentList">
            {ragDocuments.length === 0 ? (
              <div className="similarCaseEmpty">
                <BookOpen size={20} />
                <span>리포트에 포함된 RAG 문서가 없습니다.</span>
              </div>
            ) : reportRagDocuments.map((item) => {
              const retrievalMode = ragRetrievalMode(item.document);
              return (
                <article className={`ragDocumentCard ${item.scoreTone}`} key={item.document.document_id}>
                  <div className="ragDocumentRail">
                    <span className="ragDocumentRank">#{item.rank}</span>
                    <b className={`ragDocumentScore ${item.scoreTone}`}>{formatMetric(item.document.relevance)}</b>
                    <div className="ragDocumentScoreBar" aria-label={`관련도 ${formatMetric(item.document.relevance)}`}>
                      <span style={{width: `${metricWidth(item.document.relevance)}%`}} />
                    </div>
                  </div>
                  <div className="ragDocumentBody">
                    <div className="ragDocumentTitleRow">
                      <div className="ragDocumentTitleStack">
                        <div className="ragDocumentPills">
                          <span className="ragDocumentType">{item.sourceLabel}</span>
                          <span className={`ragModePill ${retrievalMode}`}>{item.retrievalLabel}</span>
                        </div>
                        <strong>{ragDocumentTitle(item.document)}</strong>
                      </div>
                      <small>관련도 {formatMetric(item.document.relevance)}</small>
                    </div>
                    <p className="ragDocumentSummaryText">{item.summary}</p>
                    {item.facts.length > 0 ? (
                      <dl className="ragDocumentFactGrid">
                        {item.facts.map((fact) => (
                          <div key={`${item.document.document_id}-${fact.label}`}>
                            <dt>{fact.label}</dt>
                            <dd>{fact.value}</dd>
                          </div>
                        ))}
                        {item.hiddenFactCount > 0 ? (
                          <div className="ragDocumentMoreFacts">
                            <dt>추가</dt>
                            <dd>{item.hiddenFactCount}개 항목</dd>
                          </div>
                        ) : null}
                      </dl>
                    ) : null}
                    <div className="ragDocumentFooter">
                      <small>출처</small>
                      <span title={item.document.source}>{item.document.source}</span>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="reportSection">
          <div className="reportSectionHeader">
            <span>04</span>
            <div>
              <h5>참조 사례</h5>
              <p>현재 판정과 비교한 과거 사례를 첨부 근거로 함께 보관합니다.</p>
            </div>
          </div>
          <div className="reportReferenceCases">
            <SimilarCaseBoard cases={referenceCases.slice(0, 2)} title="리포트 참조 사례" onOpenCase={props.onOpenCase} />
          </div>
        </section>
      </article>
    </div>
  );
}

function EvidenceLinkList(props: { readonly items: readonly ReportEvidenceLink[] }) {
  if (props.items.length === 0) {
    return (
      <div className="evidenceLinkList empty">
        <span>연결 근거 없음</span>
      </div>
    );
  }

  return (
    <div className="evidenceLinkList">
      {props.items.map((item) => (
        <div className={`evidenceLink ${item.kind}`} key={`${item.kind}-${item.title}-${item.reason}`}>
          <span>{item.kind === "document" ? "문서" : "사례"}</span>
          <strong>{item.title}</strong>
          <small>{item.reason} · {formatMetric(item.score)}</small>
        </div>
      ))}
    </div>
  );
}

function RagAdminPanel(props: {
  readonly onRefreshRuntime: () => Promise<void>;
  readonly runtimeStatus: ModelRuntimeStatus | null;
}) {
  const [panelStatus, setPanelStatus] = useState<RagPanelStatus>("loading");
  const [runtimeRefreshStatus, setRuntimeRefreshStatus] = useState<RuntimeRefreshStatus>("idle");
  const [status, setStatus] = useState<RagStatusResponse | null>(null);
  const [documents, setDocuments] = useState<readonly RagDocumentListItem[]>([]);
  const [queryLogs, setQueryLogs] = useState<readonly RagQueryLogItem[]>([]);
  const [sourceType, setSourceType] = useState("all");
  const [searchQuery, setSearchQuery] = useState("HFCT 코로나 방전 근거");
  const [searchTopK, setSearchTopK] = useState(3);
  const [searchResults, setSearchResults] = useState<readonly RagDocument[]>([]);
  const [searchFilters, setSearchFilters] = useState<readonly RagAppliedFilter[]>([]);
  const [searchRetrievalMode, setSearchRetrievalMode] = useState<string | null>(null);
  const [searchResultCount, setSearchResultCount] = useState(0);
  const [lastSearchQuery, setLastSearchQuery] = useState("");
  const [searchError, setSearchError] = useState<string | null>(null);
  const [datasetLimit, setDatasetLimit] = useState("");
  const [reindexResult, setReindexResult] = useState<RagReindexResponse | null>(null);
  const [documentDetail, setDocumentDetail] = useState<RagDocumentDetailResponse | null>(null);
  const [documentDetailStatus, setDocumentDetailStatus] = useState<RagDocumentDetailStatus>("idle");
  const documentSampleLimit = RAG_ADMIN_LIST_LIMIT;

  useEffect(() => {
    void refreshRagAdmin();
  }, [sourceType]);

  async function refreshRagAdmin(): Promise<void> {
    setPanelStatus("loading");
    try {
      const [nextStatus, nextDocuments, nextLogs] = await Promise.all([
        fetchRagStatus(),
        fetchRagDocuments({sourceType, limit: documentSampleLimit}),
        fetchRagQueryLogs(RAG_ADMIN_LIST_LIMIT),
      ]);
      setStatus(nextStatus);
      setDocuments(nextDocuments.items.slice(0, documentSampleLimit));
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
    setLastSearchQuery(query);
    setSearchError(null);
    try {
      const result = await searchRagDocuments({query, topK: boundedRagTopK(searchTopK)});
      const nextLogs = await fetchRagQueryLogs(RAG_ADMIN_LIST_LIMIT);
      setSearchResults(result.documents);
      setSearchFilters(result.applied_filters);
      setSearchRetrievalMode(result.retrieval_mode);
      setSearchResultCount(result.result_count);
      setSearchError(result.error);
      setQueryLogs(nextLogs.items.slice(0, RAG_ADMIN_LIST_LIMIT));
      setPanelStatus("idle");
    } catch {
      setSearchResults([]);
      setSearchFilters([]);
      setSearchRetrievalMode(null);
      setSearchResultCount(0);
      setSearchError("RAG 관리 API 호출에 실패했습니다.");
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

  async function openDocumentDetail(documentKey: string): Promise<void> {
    setDocumentDetailStatus("loading");
    try {
      setDocumentDetail(await fetchRagDocumentDetail(documentKey));
      setDocumentDetailStatus("idle");
    } catch {
      setDocumentDetail(null);
      setDocumentDetailStatus("failed");
    }
  }

  async function handleRuntimeRefresh(): Promise<void> {
    setRuntimeRefreshStatus("loading");
    try {
      await props.onRefreshRuntime();
      setRuntimeRefreshStatus("idle");
    } catch {
      setRuntimeRefreshStatus("failed");
    }
  }

  const sourceTypes = ragSourceTypes(status);
  return (
    <div className="ragAdminPanel">
      <LlmRagReporterPanel
        refreshStatus={runtimeRefreshStatus}
        status={props.runtimeStatus}
        onRefresh={() => void handleRuntimeRefresh()}
      />

      <section className="ragStatusGrid">
        <RuntimeStatusItem label="RAG 상태" value={status === null ? "확인 중" : status.ready ? "준비됨" : "확인 필요"} tone={status?.ready ? "ready" : "warn"} />
        <RuntimeStatusItem label="DB 연결" value={status === null ? "확인 중" : status.database_connected ? "연결됨" : "끊김"} tone={status?.database_connected ? "ready" : "warn"} />
        <RuntimeStatusItem label="문서 / Chunk" value={status === null ? "0 / 0" : `${status.document_count} / ${status.chunk_count}`} tone="neutral" />
        <RuntimeStatusItem label="마지막 색인" value={status?.last_indexed_at ? formatDate(status.last_indexed_at) : "없음"} tone={status?.last_indexed_at ? "neutral" : "warn"} />
      </section>
      <RagIndexHealthStrip status={status} />

      <section className="ragIndexConsole">
        <article className="ragIndexCard browse">
          <div className="ragIndexHeader">
            <div>
              <span>조회</span>
              <strong>색인 문서 보기</strong>
              <p>이미 RAG DB에 들어간 근거 문서를 출처별로 확인합니다. 진단 설정이나 검색 모드는 바뀌지 않습니다.</p>
            </div>
            <button className="downloadButton" type="button" onClick={() => void refreshRagAdmin()}>
              <RefreshCw size={16} />
              새로고침
            </button>
          </div>
          <SourceTypeTabs selected={sourceType} sourceTypes={sourceTypes} status={status} onSelect={setSourceType} />
        </article>

        <article className="ragIndexCard rebuild">
          <div className="ragIndexHeader">
            <div>
              <span>고급 관리</span>
              <strong>RAG 인덱스 재구축</strong>
              <p>원본 판정 기준, 운영 절차, 과거 사례를 다시 읽어 embedding과 chunk를 업데이트합니다.</p>
            </div>
          </div>
          <div className="ragReindexControls">
            <label className="field">
              <span>재색인할 과거 사례 수</span>
              <input inputMode="numeric" max={50000} min={1} placeholder="전체" value={datasetLimit} onChange={(event) => setDatasetLimit(event.currentTarget.value)} />
              <small className="fieldHint">비워두면 전체 과거 사례를 재색인합니다. 개발 검증 시에만 숫자로 제한하세요.</small>
            </label>
            <button className="primaryButton" type="button" disabled={panelStatus === "reindexing"} onClick={() => void handleReindex()}>
              {panelStatus === "reindexing" ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
              재색인
            </button>
          </div>
          <p className="ragIndexFootnote">문서 내용, 데이터셋 manifest, embedding 설정이 바뀐 경우에만 실행하는 작업입니다.</p>
        </article>
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
        <RagSearchResults
          documents={searchResults}
          error={searchError}
          filters={searchFilters}
          query={lastSearchQuery}
          resultCount={searchResultCount}
          retrievalMode={searchRetrievalMode}
          onOpenDocument={(documentKey) => void openDocumentDetail(documentKey)}
        />
        <RagDocumentTable
          documents={documents}
          sampleLimit={documentSampleLimit}
          searchError={searchError}
          searchDocuments={searchResults}
          searchQuery={lastSearchQuery}
          onOpenDocument={(documentKey) => void openDocumentDetail(documentKey)}
        />
        <RagQueryLogList logs={queryLogs} />
      </section>

      {panelStatus === "failed" ? (
        <div className="similarCaseEmpty ragError">
          <AlertTriangle size={20} />
          <span>RAG 관리 API 호출에 실패했습니다.</span>
        </div>
      ) : null}
      <RagDocumentDetailModal
        detail={documentDetail}
        status={documentDetailStatus}
        onClose={() => {
          setDocumentDetail(null);
          setDocumentDetailStatus("idle");
        }}
      />
    </div>
  );
}

function LlmRagReporterPanel(props: {
  readonly onRefresh: () => void;
  readonly refreshStatus: RuntimeRefreshStatus;
  readonly status: ModelRuntimeStatus | null;
}) {
  const status = props.status;
  const ready = status?.llm_rag_ready ?? false;
  return (
    <section className={`llmRagReporter ${ready ? "ready" : "fallback"}`}>
      <div className="llmRagHeader">
        <div>
          <span>OpenRouter RAG 보조</span>
          <strong>{llmRagStateLabel(status)}</strong>
          <p>{llmRagStateDetail(status)}</p>
        </div>
        <button className="downloadButton" type="button" disabled={props.refreshStatus === "loading"} onClick={props.onRefresh}>
          {props.refreshStatus === "loading" ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
          상태 확인
        </button>
      </div>

      <div className="llmRagStatusGrid">
        <RuntimeStatusItem label="Provider" value={status?.llm_rag_provider ?? "확인 중"} tone={ready ? "ready" : "warn"} />
        <RuntimeStatusItem label="LLM 모델" value={status?.llm_rag_model ?? "미설정"} tone={ready ? "ready" : "neutral"} />
        <RuntimeStatusItem label="Active adapter" value={status?.llm_rag_adapter ?? "확인 중"} tone="neutral" />
        <RuntimeStatusItem label="Base VLM" value={status?.vlm_model ?? "확인 중"} tone={status?.vlm_ready ? "ready" : "warn"} />
      </div>

      <div className="llmRagFlow" aria-label="LLM RAG 처리 흐름">
        {llmRagFlowSteps(status).map((step, index) => (
          <article className={`llmRagFlowStep ${step.state}`} key={step.label}>
            <span>{index + 1}</span>
            <div>
              <strong>{step.title}</strong>
              <small>{step.label}</small>
              <p>{step.detail}</p>
            </div>
          </article>
        ))}
      </div>

      <div className="llmRagReviewGrid">
        <article>
          <span>검수 결과</span>
          <strong>{ready ? "OpenRouter RAG 보조 활성" : "OpenRouter 비활성"}</strong>
          <p>{ready ? "RAG 챗과 보조 설명 생성에 OpenRouter를 사용합니다." : "진단 VLM 어댑터와 mock 경로만 사용합니다."}</p>
        </article>
        <article>
          <span>전제 조건</span>
          <strong>{ready ? "API 설정 확인됨" : "OpenRouter 설정 필요"}</strong>
          <p>{status?.llm_rag_error ?? "OPENROUTER_API_KEY와 RAG 색인 상태를 함께 확인합니다."}</p>
        </article>
      </div>

      {props.refreshStatus === "failed" ? <p className="ragError">OpenRouter RAG 보조 상태를 갱신하지 못했습니다.</p> : null}
    </section>
  );
}

function RagIndexHealthStrip(props: { readonly status: RagStatusResponse | null }) {
  const status = props.status;
  const missingSummary = ragMetadataMissingSummary(status);
  return (
    <section className="ragIndexHealthStrip">
      <div>
        <span>pgvector</span>
        <strong>{status?.vector_extension ?? "확인 중"}</strong>
      </div>
      <div>
        <span>Embedding</span>
        <strong>{status === null ? "확인 중" : `${status.embedding_model} · ${status.vector_dim}d`}</strong>
      </div>
      <div>
        <span>Query log</span>
        <strong>{status === null ? "0" : status.query_log_count.toLocaleString()}</strong>
      </div>
      <div className={missingSummary.tone}>
        <span>Metadata 누락</span>
        <strong>{missingSummary.text}</strong>
      </div>
    </section>
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
      <span>문서 출처 필터</span>
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
            <small>{sourceTypeCount(sourceType, props.status).toLocaleString()}개 문서</small>
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

function SimilarCaseLimitControl(props: {
  readonly max: number;
  readonly onChange: (value: number) => void;
  readonly value: number;
}) {
  const value = boundedSimilarCaseLimit(props.value, props.max);
  return (
    <div className="topKControl similarCaseLimitControl">
      <span>표시 개수</span>
      <div>
        <button
          aria-label="유사 사례 표시 개수 감소"
          disabled={value <= 1}
          type="button"
          onClick={() => props.onChange(boundedSimilarCaseLimit(value - 1, props.max))}
        >
          -
        </button>
        <input
          aria-label="유사 사례 표시 개수"
          inputMode="numeric"
          max={Math.max(1, props.max)}
          min={1}
          type="number"
          value={value}
          onChange={(event) => props.onChange(boundedSimilarCaseLimit(Number(event.currentTarget.value), props.max))}
        />
        <button
          aria-label="유사 사례 표시 개수 증가"
          disabled={value >= props.max}
          type="button"
          onClick={() => props.onChange(boundedSimilarCaseLimit(value + 1, props.max))}
        >
          +
        </button>
      </div>
    </div>
  );
}

function RagSearchResults(props: {
  readonly documents: readonly RagDocument[];
  readonly error: string | null;
  readonly filters: readonly RagAppliedFilter[];
  readonly onOpenDocument: (documentKey: string) => void;
  readonly query: string;
  readonly resultCount: number;
  readonly retrievalMode: string | null;
}) {
  const hasSearched = props.query.trim().length > 0;
  const resultTitle = ragSearchResultTitle(props.documents, props.error, hasSearched);
  return (
    <section className="ragAdminCard ragSearchResultPanel">
      <div className="ragCardHeader">
        <div>
          <div className="sectionTitle">검색 결과</div>
          <strong>{resultTitle}</strong>
        </div>
        <span>{props.retrievalMode === null ? "검색 대기" : retrievalModeLabel(props.retrievalMode)}</span>
      </div>
      <RagAppliedFilterPanel
        filters={props.filters}
        hasSearched={hasSearched}
        resultCount={props.resultCount}
      />
      {props.error !== null ? (
        <div className="similarCaseEmpty ragError">
          <AlertTriangle size={20} />
          <span>{props.error}</span>
        </div>
      ) : props.documents.length === 0 ? (
        <div className="similarCaseEmpty">
          <Search size={20} />
          <span>{hasSearched ? "현재 질의와 일치하는 RAG 문서가 없습니다." : "검색을 실행하면 상위 문서가 표시됩니다."}</span>
        </div>
      ) : (
        <div className="ragResultList">
          {props.documents.map((document) => (
            <RagSearchResultCard document={document} key={document.document_id} onOpenDocument={props.onOpenDocument} />
          ))}
        </div>
      )}
    </section>
  );
}

function RagAppliedFilterPanel(props: {
  readonly filters: readonly RagAppliedFilter[];
  readonly hasSearched: boolean;
  readonly resultCount: number;
}) {
  const filters = props.filters;
  return (
    <div className={`ragAppliedFilters ${filters.length === 0 ? "empty" : ""}`}>
      <div>
        <span>적용된 필터</span>
        <strong>{props.hasSearched ? `${props.resultCount}개 결과` : "검색 전"}</strong>
      </div>
      {filters.length === 0 ? (
        <p>{props.hasSearched ? "구조화 조건 없이 의미 검색만 실행했습니다." : "검색하면 인식된 조건이 표시됩니다."}</p>
      ) : (
        <div>
          {filters.map((filter) => (
            <span className="ragFilterChip" key={`${filter.key}-${filter.value}`}>
              <small>{filter.label}</small>
              <b>{filter.value}</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ragSearchResultTitle(
  documents: readonly RagDocument[],
  error: string | null,
  hasSearched: boolean,
): string {
  if (error !== null) {
    return "검색 오류";
  }
  if (documents.length > 0) {
    return `${documents.length}개 질문 근거`;
  }
  return hasSearched ? "검색 결과 없음" : "검색 대기";
}

function RagSearchResultCard(props: {
  readonly document: RagDocument;
  readonly onOpenDocument?: (documentKey: string) => void;
}) {
  const document = props.document;
  const facts = ragDocumentFacts(document);
  const documentKey = documentKeyFromRagDocument(document);
  return (
    <article className="ragResultCard">
      <div className="ragResultTopline">
        <div>
          <span className={`ragSourcePill ${document.source_type ?? "unknown"}`}>{sourceTypeLabel(document.source_type ?? undefined)}</span>
          <span className={`ragModePill ${ragRetrievalMode(document)}`}>{retrievalModeLabel(ragRetrievalMode(document))}</span>
          <strong>{ragDocumentTitle(document)}</strong>
          <small>{document.source}</small>
        </div>
        <b className={ragScoreTone(document.relevance)}>{formatMetric(document.relevance)}</b>
      </div>

      <div className="ragFactGrid">
        {facts.map((fact) => (
          <span className="ragFactChip" key={`${document.document_id}-${fact.label}`}>
            <small>{fact.label}</small>
            <strong>{fact.value}</strong>
          </span>
        ))}
      </div>

      <p className="ragResultSummary">{ragDocumentSummary(document, facts)}</p>
      {props.onOpenDocument === undefined ? null : (
        <button className="downloadButton compact" type="button" onClick={() => props.onOpenDocument?.(documentKey)}>
          <FileSearch size={14} />
          자세히 보기
        </button>
      )}
      <details className="ragRawDetails">
        <summary>검색 chunk 보기</summary>
        <pre>{document.excerpt}</pre>
      </details>
    </article>
  );
}

function RagDocumentDetailModal(props: {
  readonly detail: RagDocumentDetailResponse | null;
  readonly onClose: () => void;
  readonly status: RagDocumentDetailStatus;
}) {
  if (props.detail === null && props.status !== "loading" && props.status !== "failed") {
    return null;
  }
  const detail = props.detail;
  const metadataEntries = detail === null ? [] : Object.entries(detail.metadata).filter(([, value]) => value !== null && value !== "");
  const detailFacts = detail === null ? [] : ragDocumentDetailFacts(detail);
  return (
    <div className="modalOverlay" role="presentation" onMouseDown={props.onClose}>
      <section
        aria-label="RAG 색인 문서 상세"
        aria-modal="true"
        className="similarCaseModal ragDocumentModal"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modalHeader">
          <div>
            <span>{detail === null ? "RAG INDEX SOURCE" : sourceTypeLabel(detail.source_type)}</span>
            <strong>{detail === null ? "문서 불러오는 중" : ragDetailTitle(detail)}</strong>
          </div>
          <button className="iconOnlyButton" type="button" aria-label="닫기" onClick={props.onClose}>
            <X size={18} />
          </button>
        </div>

        {props.status === "loading" ? (
          <div className="modalLoading">
            <Loader2 className="spin" size={16} />
            문서 원문을 불러오는 중입니다.
          </div>
        ) : null}
        {props.status === "failed" ? <p className="ragError">문서 상세를 불러오지 못했습니다.</p> : null}
        {detail === null ? null : (
          <div className="ragDocumentDetailBody">
            <div className="ragDocumentHero">
              <div>
                <span className={`ragSourcePill ${detail.source_type}`}>{sourceTypeLabel(detail.source_type)}</span>
                <strong>{ragDetailTitle(detail)}</strong>
                <p>{ragDetailLead(detail)}</p>
              </div>
              <dl>
                <div><dt>Chunk</dt><dd>{detail.chunks.length}</dd></div>
                <div><dt>Updated</dt><dd>{formatDate(detail.updated_at)}</dd></div>
              </dl>
            </div>

            {detailFacts.length === 0 ? null : (
              <div className="ragFactGrid">
                {detailFacts.map((fact) => (
                  <span className="ragFactChip" key={fact.label}>
                    <small>{fact.label}</small>
                    <strong>{fact.value}</strong>
                  </span>
                ))}
              </div>
            )}

            <div className="ragReader">
              <div className="sectionTitle">문서 내용</div>
              {detail.chunks.length === 0 ? (
                <p>저장된 chunk 원문이 없습니다.</p>
              ) : (
                detail.chunks.map((chunk) => (
                  <section key={chunk.chunk_key}>
                    <span>#{chunk.chunk_index + 1}</span>
                    <RagReadableText text={chunk.text} />
                  </section>
                ))
              )}
            </div>

            <details className="ragRawDetails">
              <summary>원본 key / metadata 보기</summary>
              <pre>{JSON.stringify({
                document_key: detail.document_key,
                source_path: detail.source_path,
                metadata: Object.fromEntries(metadataEntries),
              }, null, 2)}</pre>
            </details>
          </div>
        )}
      </section>
    </div>
  );
}

function RagReadableText(props: { readonly text: string }) {
  const rows = props.text.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  return (
    <dl className="ragReadableText">
      {rows.map((row) => {
        const [rawKey, ...rest] = row.split("=");
        const key = rawKey ?? row;
        const value = rest.join("=");
        if (value === "") {
          return <p key={row}>{row}</p>;
        }
        return (
          <div key={row}>
            <dt>{ragReadableLabel(key)}</dt>
            <dd>{value}</dd>
          </div>
        );
      })}
    </dl>
  );
}

function RagDocumentTable(props: {
  readonly documents: readonly RagDocumentListItem[];
  readonly onOpenDocument: (documentKey: string) => void;
  readonly sampleLimit: number;
  readonly searchError: string | null;
  readonly searchDocuments: readonly RagDocument[];
  readonly searchQuery: string;
}) {
  const hasSearch = props.searchQuery.trim().length > 0 || props.searchError !== null;
  const hasSearchDocuments = props.searchDocuments.length > 0;
  return (
    <section className="ragAdminCard">
      <div className="ragCardHeader">
        <div>
          <div className="sectionTitle">{hasSearch ? "검색 결과 문서" : "최근 색인 문서"}</div>
          <strong>{hasSearch ? ragDocumentTableTitle(props.searchDocuments, props.searchError) : "검색 전 색인 샘플"}</strong>
        </div>
        <span>{hasSearch ? `${props.searchDocuments.length}개` : `최근 ${props.sampleLimit}개`}</span>
      </div>
      {hasSearch ? (
        <RagSearchDocumentRows documents={props.searchDocuments} onOpenDocument={props.onOpenDocument} />
      ) : (
        <RagIndexedDocumentRows documents={props.documents} onOpenDocument={props.onOpenDocument} />
      )}
    </section>
  );
}

function ragDocumentTableTitle(documents: readonly RagDocument[], error: string | null): string {
  if (error !== null) {
    return "검색 오류";
  }
  return documents.length > 0 ? "현재 질의 상위 결과" : "해당 질의 결과 없음";
}

function RagSearchDocumentRows(props: {
  readonly documents: readonly RagDocument[];
  readonly onOpenDocument: (documentKey: string) => void;
}) {
  return (
    <div className="compactTable">
      {props.documents.map((document) => {
        const documentKey = documentKeyFromRagDocument(document);
        return (
          <article className="compactRow" key={document.document_id}>
            <div>
              <strong>{ragDocumentTitle(document)}</strong>
              <span>{ragDocumentSummary(document, ragDocumentFacts(document))}</span>
            </div>
            <b>{formatMetric(document.relevance)}</b>
            <small className={`ragModePill ${ragRetrievalMode(document)}`}>{retrievalModeLabel(ragRetrievalMode(document))}</small>
            <small>{sourceTypeLabel(document.source_type ?? undefined)}</small>
            <button className="iconOnlyButton" type="button" aria-label={`${document.title} 자세히 보기`} onClick={() => props.onOpenDocument(documentKey)}>
              <FileSearch size={15} />
            </button>
          </article>
        );
      })}
      {props.documents.length === 0 ? (
        <div className="similarCaseEmpty">
          <Search size={20} />
          <span>현재 질의에 해당하는 RAG 문서가 없습니다.</span>
        </div>
      ) : null}
    </div>
  );
}

function RagIndexedDocumentRows(props: {
  readonly documents: readonly RagDocumentListItem[];
  readonly onOpenDocument: (documentKey: string) => void;
}) {
  return (
      <div className="compactTable">
        {props.documents.map((document) => (
          <article className="compactRow" key={document.document_key}>
            <div>
              <strong>{document.title}</strong>
              <span>{document.source_path ?? document.document_key}</span>
            </div>
            <b>{sourceTypeLabel(document.source_type)}</b>
            <small>{document.chunk_count} chunks</small>
            <button className="iconOnlyButton" type="button" aria-label={`${document.title} 자세히 보기`} onClick={() => props.onOpenDocument(document.document_key)}>
              <FileSearch size={15} />
            </button>
          </article>
        ))}
        {props.documents.length === 0 ? (
          <div className="similarCaseEmpty">
            <BookOpen size={20} />
            <span>등록된 RAG 문서가 없습니다.</span>
          </div>
        ) : null}
      </div>
  );
}

function RagQueryLogList(props: { readonly logs: readonly RagQueryLogItem[] }) {
  return (
    <section className="ragAdminCard">
      <div className="ragCardHeader">
        <div>
          <div className="sectionTitle">최근 질의 로그</div>
          <strong>RAG 호출 이력</strong>
        </div>
        <span>TOP 5</span>
      </div>
      <div className="ragLogList">
        {props.logs.map((log) => (
          <RagQueryLogRow key={log.id} log={log} />
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

function RagQueryLogRow(props: { readonly log: RagQueryLogItem }) {
  const log = props.log;
  const metadata = log.query_metadata;
  return (
    <article className="ragLogRow">
      <div className="ragLogMeta">
        <span>{log.diagnosis_id ?? "standalone"}</span>
        <b>{formatDate(log.created_at)}</b>
      </div>
      <strong>{compactText(log.query_text, 150)}</strong>
      <div className="ragLogFooter">
        <small>{log.retrieved_chunks.length} chunks</small>
        <small>{queryMetadataSummary(metadata)}</small>
      </div>
    </article>
  );
}

function RagChatPanel(props: { readonly runtimeStatus: ModelRuntimeStatus | null }) {
  const [question, setQuestion] = useState("HFCT 코로나 방전 근거");
  const [topK, setTopK] = useState(3);
  const [status, setStatus] = useState<RagChatStatus>("idle");
  const [turns, setTurns] = useState<readonly RagChatTurn[]>([]);
  const [documentDetail, setDocumentDetail] = useState<RagDocumentDetailResponse | null>(null);
  const [documentDetailStatus, setDocumentDetailStatus] = useState<RagDocumentDetailStatus>("idle");
  const ready = props.runtimeStatus?.llm_rag_ready ?? false;
  const setupError = props.runtimeStatus?.llm_rag_error ?? null;

  async function handleAsk(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (cleanQuestion.length === 0 || status === "asking") {
      return;
    }
    const nextUserTurn: RagChatTurn = {role: "user", content: cleanQuestion};
    const chatHistory = [...turns, nextUserTurn].map(({role, content}) => ({role, content}));
    setTurns((current) => [...current, nextUserTurn]);
    setQuestion("");
    setStatus("asking");
    try {
      const response = await askRagChat({
        messages: chatHistory.slice(0, -1),
        question: cleanQuestion,
        topK: boundedRagTopK(topK),
      });
      const assistantTurn: RagChatTurn = {
        role: "assistant",
        content: response.ready ? response.answer : response.error ?? "RAG 챗을 사용할 수 없습니다.",
        answerMode: response.answer_mode,
        documents: response.documents,
        error: response.error,
        model: response.model,
      };
      setTurns((current) => [...current, assistantTurn]);
      setStatus("idle");
    } catch {
      setTurns((current) => [
        ...current,
        {role: "assistant", content: "RAG 챗 API 호출에 실패했습니다.", error: "request_failed"},
      ]);
      setStatus("failed");
    }
  }

  async function openDocumentDetail(documentKey: string): Promise<void> {
    setDocumentDetailStatus("loading");
    try {
      setDocumentDetail(await fetchRagDocumentDetail(documentKey));
      setDocumentDetailStatus("idle");
    } catch {
      setDocumentDetail(null);
      setDocumentDetailStatus("failed");
    }
  }

  return (
    <div className="ragChatPanel">
      <section className={`ragChatStatus ${ready ? "ready" : "warn"}`}>
        <div>
          <span>OpenRouter RAG Chat</span>
          <strong>{ready ? "대화형 RAG 사용 가능" : "API 키 설정 대기"}</strong>
          <p>{ready ? `${props.runtimeStatus?.llm_rag_model ?? "OpenRouter 모델"}로 RAG 근거 답변을 생성합니다.` : setupError ?? "OPENROUTER_API_KEY를 설정하면 활성화됩니다."}</p>
        </div>
        <div className="ragChatStatusControls">
          <span className={`miniStatus ${ready ? "ok" : "warn"}`}>{ready ? "Ready" : "Setup"}</span>
          <TopKControl value={topK} onChange={setTopK} />
        </div>
      </section>

      <section className="ragChatShell">
        <div className="ragChatTranscript" aria-live="polite">
          {turns.length === 0 ? (
            <div className="ragChatEmpty">
              <MessageSquare size={28} />
              <strong>RAG 근거 기반 질문을 입력하세요.</strong>
              <p>예: 보이드 방전 판단 기준, HFCT 코로나 방전 근거, 특정 설비 사례 비교</p>
            </div>
          ) : (
            turns.map((turn, index) => (
              <RagChatBubble
                key={`${turn.role}-${index}`}
                turn={turn}
                onOpenDocument={(documentKey) => void openDocumentDetail(documentKey)}
              />
            ))
          )}
          {status === "asking" ? (
            <div className="ragChatBubble assistant pending">
              <Loader2 className="spin" size={16} />
              <span>RAG 문서를 검색하고 OpenRouter 답변을 생성 중입니다.</span>
            </div>
          ) : null}
        </div>

        <form className="ragChatComposer" onSubmit={(event) => void handleAsk(event)}>
          <label className="field">
            <span>질문</span>
            <textarea
              placeholder="부분방전/RAG 근거 질문을 입력하세요"
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.currentTarget.value)}
            />
          </label>
          <div className="ragChatComposerControls">
            <button className="primaryButton" type="submit" disabled={status === "asking"}>
              {status === "asking" ? <Loader2 className="spin" size={16} /> : <Send size={16} />}
              질문
            </button>
          </div>
        </form>
      </section>

      {status === "failed" ? <p className="ragError">RAG 챗 API 호출에 실패했습니다.</p> : null}
      <RagDocumentDetailModal
        detail={documentDetail}
        status={documentDetailStatus}
        onClose={() => {
          setDocumentDetail(null);
          setDocumentDetailStatus("idle");
        }}
      />
    </div>
  );
}

function RagChatBubble(props: {
  readonly onOpenDocument: (documentKey: string) => void;
  readonly turn: RagChatTurn;
}) {
  const turn = props.turn;
  return (
    <article className={`ragChatBubble ${turn.role} ${turn.error ? "warn" : ""}`}>
      <div>
        <div className="ragChatBubbleHeader">
          <span>{turn.role === "user" ? "사용자" : turn.model ?? "RAG 챗"}</span>
          {turn.role === "assistant" ? (
            <span className={`ragChatModeBadge ${ragChatModeClass(turn)}`}>{ragChatModeLabel(turn)}</span>
          ) : null}
        </div>
        <RagChatContent content={turn.content} role={turn.role} />
      </div>
      {turn.documents !== undefined && turn.documents.length > 0 ? (
        <details className="ragChatEvidence">
          <summary>근거 {turn.documents.length}개 보기</summary>
          <div className="ragResultList compact">
            {turn.documents.map((document) => (
              <RagSearchResultCard document={document} key={document.document_id} onOpenDocument={props.onOpenDocument} />
            ))}
          </div>
        </details>
      ) : null}
    </article>
  );
}

function ragChatModeLabel(turn: RagChatTurn): string {
  if (turn.error) {
    return "오류";
  }
  switch (turn.answerMode) {
    case "rag_evidence":
      return `근거 기반${turn.documents === undefined ? "" : ` ${turn.documents.length}건`}`;
    case "general_domain":
      return "일반 설명";
    case "diagnosis_history":
      return "진단 이력";
    case "out_of_scope":
      return "범위 밖";
    default:
      return turn.documents !== undefined && turn.documents.length > 0 ? `근거 기반 ${turn.documents.length}건` : "응답";
  }
}

function ragChatModeClass(turn: RagChatTurn): string {
  if (turn.error) {
    return "warn";
  }
  switch (turn.answerMode) {
    case "rag_evidence":
      return "evidence";
    case "general_domain":
      return "general";
    case "diagnosis_history":
      return "history";
    case "out_of_scope":
      return "scope";
    default:
      return "neutral";
  }
}

function RagChatContent(props: { readonly content: string; readonly role: RagChatMessage["role"] }) {
  if (props.role === "user") {
    return <p className="ragChatText">{props.content}</p>;
  }

  const blocks = readableChatBlocks(props.content);
  return (
    <div className="ragChatContent">
      {blocks.map((block, index) =>
        block.kind === "list" ? (
          <ul key={`list-${index}`}>
            {block.items.map((item, itemIndex) => (
              <li key={`${item}-${itemIndex}`}>{item}</li>
            ))}
          </ul>
        ) : block.kind === "heading" ? (
          <strong className="ragChatSectionTitle" key={`heading-${index}`}>
            {block.text}
          </strong>
        ) : (
          <p key={`paragraph-${index}`}>{block.text}</p>
        ),
      )}
    </div>
  );
}

function readableChatBlocks(content: string): RagChatContentBlock[] {
  const clean = content.trim();
  if (!clean) {
    return [{kind: "paragraph", text: ""}];
  }

  const lines = clean.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length > 1) {
    return structuredChatBlocks(lines);
  }

  const structuredSingleLine = structuredLabelLine(clean);
  if (structuredSingleLine !== null) {
    return structuredSingleLine;
  }
  if (clean.length < 150) {
    return [{kind: "paragraph", text: clean}];
  }

  const sentences = clean.split(/(?<=[.!?])\s+/).map((sentence) => sentence.trim()).filter(Boolean);
  if (sentences.length < 2) {
    return [{kind: "paragraph", text: clean}];
  }
  const [leadSentence, ...detailSentences] = sentences;
  if (leadSentence === undefined) {
    return [{kind: "paragraph", text: clean}];
  }

  return [
    {kind: "paragraph", text: leadSentence},
    {kind: "list", items: detailSentences},
  ];
}

function structuredChatBlocks(lines: readonly string[]): RagChatContentBlock[] {
  const blocks: RagChatContentBlock[] = [];
  let pendingList: string[] = [];

  function flushList(): void {
    if (pendingList.length > 0) {
      blocks.push({kind: "list", items: pendingList});
      pendingList = [];
    }
  }

  for (const line of lines) {
    const bullet = line.match(/^[-*•]\s+(.+)$/);
    if (bullet?.[1] !== undefined) {
      pendingList.push(bullet[1].trim());
      continue;
    }

    const labeled = structuredLabelLine(line);
    if (labeled !== null) {
      flushList();
      blocks.push(...labeled);
      continue;
    }

    flushList();
    blocks.push({kind: "paragraph", text: line});
  }

  flushList();
  return blocks;
}

function structuredLabelLine(line: string): RagChatContentBlock[] | null {
  const match = line.match(/^(핵심 판단|판단 근거|주요 근거|참고 사례|추가 확인|권장 확인|주의|결론)\s*:\s*(.*)$/);
  if (match?.[1] === undefined || match[2] === undefined) {
    return null;
  }

  const title = match[1];
  const detail = match[2].trim();
  const inlineItems = inlineBulletItems(detail);
  if (inlineItems.length > 0) {
    return [{kind: "heading", text: title}, {kind: "list", items: inlineItems}];
  }
  return detail.length > 0
    ? [{kind: "heading", text: title}, {kind: "paragraph", text: detail}]
    : [{kind: "heading", text: title}];
}

function inlineBulletItems(text: string): string[] {
  if (!text.startsWith("- ")) {
    return [];
  }
  return text
    .split(/\s+-\s+/)
    .map((item) => item.replace(/^-\s+/, "").trim())
    .filter(Boolean);
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
  if (props.trace.events.length === 0) {
    return (
      <div className="emptyState compact">
        <Cpu size={24} />
        <strong>기록된 추적 이벤트 없음</strong>
        <span>진단은 생성됐지만 백엔드 trace 이벤트가 비어 있습니다.</span>
      </div>
    );
  }

  return (
    <div className="traceLogPanel">
      <div className="traceLogSummary">
        <span>{routeLabel[props.trace.route]}</span>
        <strong>{props.trace.events.length}개 처리 단계</strong>
        <small>행을 클릭하거나 상세 버튼을 누르면 해당 단계 payload가 열립니다.</small>
      </div>
      <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>순서</th>
            <th>시각</th>
            <th>유형</th>
            <th>단계</th>
            <th>상세</th>
          </tr>
        </thead>
        <tbody>
          {props.trace.events.map((event, index) => (
            <TraceRow
              createdAt={event.created_at}
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
    </div>
  );
}

function TraceRow(props: {
  readonly createdAt: string | undefined;
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
        <td>{props.createdAt === undefined ? "없음" : formatDate(props.createdAt)}</td>
        <td>{props.kind}</td>
        <td><strong>{props.name}</strong></td>
        <td><button className="traceToggleButton" type="button">{props.expanded ? "닫기" : "보기"}</button></td>
      </tr>
      {props.expanded ? (
        <tr>
          <td colSpan={5}>
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

function EvidencePanel(props: { readonly onOpenCase: (item: SimilarCase) => void; readonly trace: TraceResponse | null }) {
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
        <PrpdEvidenceViewer imageUrl={inputArtifacts.prpdImageUrl} similarCases={similarCases} onOpenCase={props.onOpenCase} />
        <SignalEvidencePanel summary={inputArtifacts.signalSummary} trace={props.trace} />
        <ModelAgreementPanel signals={modelSignals} />
      </section>
      <section className="evidenceDetailSection">
        <div className="evidenceSectionHeader">
          <div>
            <span>Trace evidence</span>
            <strong>단계별 근거</strong>
          </div>
          <b>{rows.filter((row) => row.event !== undefined).length}/{rows.length}</b>
        </div>
        <div className="evidenceGrid">
          {rows.map((row) => {
            const payload = evidencePayload(row.event);
            return (
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
                {payload !== null ? (
                  <details className="evidencePayloadDetails">
                    <summary>상세 데이터</summary>
                    <pre className="evidenceSummary">{JSON.stringify(payload, null, 2)}</pre>
                  </details>
                ) : null}
              </article>
            );
          })}
        </div>
      </section>
      <div className="evidenceCaseBand">
        <SimilarCaseBoard cases={similarCases} title="데이터셋 유사 사례" onOpenCase={props.onOpenCase} />
      </div>
    </div>
  );
}

function PrpdEvidenceViewer(props: {
  readonly imageUrl: string | null;
  readonly onOpenCase: (item: SimilarCase) => void;
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
          <button className="sideBySideCase" key={item.sample_id} type="button" onClick={() => props.onOpenCase(item)}>
            <span>{formatMetric(item.similarity)}</span>
            <img alt={`${item.sample_id} 유사 사례`} src={apiAssetUrl(item.image_url)} />
            <strong>{item.label_name}</strong>
          </button>
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
            <small title={signal.detail}>{signal.detail}</small>
          </div>
        ))}
      </div>
    </article>
  );
}

function CurrentSimilarCasesPanel(props: {
  readonly cases: readonly SimilarCase[];
  readonly hasTrace: boolean;
  readonly limit: number;
  readonly onLimitChange: (value: number) => void;
  readonly onOpenCase: (item: SimilarCase) => void;
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

  const visibleCases = props.cases.slice(0, boundedSimilarCaseLimit(props.limit, props.cases.length));
  return (
    <div className="currentCasePanel">
      <div className="currentCaseBrief">
        <div>
          <span>자동 매칭</span>
          <strong>{props.cases.length > 0 ? `상위 ${visibleCases.length}건 표시 / 전체 ${props.cases.length}건` : "매칭된 과거 사례 없음"}</strong>
        </div>
        {props.cases.length > 0 ? (
          <SimilarCaseLimitControl
            max={props.cases.length}
            value={visibleCases.length}
            onChange={props.onLimitChange}
          />
        ) : null}
      </div>
      <SimilarCaseBoard cases={visibleCases} title="과거 사례 비교" hideHeader onOpenCase={props.onOpenCase} />
    </div>
  );
}

function SimilarCaseBoard(props: {
  readonly cases: readonly SimilarCase[];
  readonly headerControl?: ReactNode;
  readonly hideHeader?: boolean;
  readonly onOpenCase?: (item: SimilarCase) => void;
  readonly subtitle?: string;
  readonly title: string;
}) {
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
            <strong>{props.subtitle ?? `상위 ${props.cases.length}건 현장 참조`}</strong>
          </div>
          {props.headerControl ?? <span>최고 유사도 {formatMetric(topCase.similarity)}</span>}
        </div>
      )}
      <div className="similarCaseGrid">
        {props.cases.map((item, index) => {
          const embeddingSimilarity = metadataNumber(item, "embedding_similarity");
          const rankSummary = similarCaseRankSummary(item, index);
          return (
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
                <div className="caseRankSummary">
                  <strong>{rankSummary.title}</strong>
                  <span>{rankSummary.detail}</span>
                </div>
                <SimilarityComponentBars item={item} compact />
                <dl>
                  <dt>유사도</dt><dd>{formatMetric(item.similarity)}</dd>
                  <dt>매칭</dt><dd><span className="caseRetrieverPill">{retrieverModeLabel(item)}</span></dd>
                  <dt>파형 피크</dt><dd>{metadataValue(item, "max_discharge_value")}</dd>
                  {embeddingSimilarity === null ? null : (
                    <>
                      <dt>Embedding</dt><dd>{formatMetric(embeddingSimilarity)}</dd>
                    </>
                  )}
                </dl>
                <p>{item.reason}</p>
                {props.onOpenCase === undefined ? null : (
                  <button className="caseOpenButton" type="button" onClick={() => props.onOpenCase?.(item)}>
                    <FileSearch size={14} />
                    사례 상세
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SimilarCaseDetailModal(props: {
  readonly currentArtifacts: InputArtifactEvidence | null;
  readonly item: SimilarCase | null;
  readonly onClose: () => void;
  readonly status: SimilarCaseDetailStatus;
  readonly trace: TraceResponse | null;
}) {
  if (props.item === null) {
    return null;
  }
  const item = props.item;
  const comparisonRows = caseComparisonRows(item, props.trace, props.currentArtifacts);
  const comparisonPoints = caseComparisonPoints(item, comparisonRows);
  return (
    <div className="modalOverlay" role="presentation" onMouseDown={props.onClose}>
      <section
        aria-label={`${item.sample_id} 유사 사례 상세`}
        aria-modal="true"
        className="similarCaseModal"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modalHeader">
          <div>
            <span>유사 사례 상세</span>
            <strong>{item.label_name}</strong>
          </div>
          <button className="iconOnlyButton" type="button" aria-label="닫기" onClick={props.onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modalBody">
          <div className="modalComparisonStack">
            <div className="modalPrpdComparison">
              <EvidenceImagePanel imageUrl={props.currentArtifacts?.prpdImageUrl ?? null} title="현재 점검 PRPD" />
              <EvidenceImagePanel imageUrl={item.image_url} title="유사 사례 PRPD" />
            </div>
            <div className="modalWaveformComparison">
              <WaveformPreview csvUrl={props.currentArtifacts?.timeseriesCsvUrl ?? null} title="현재 점검 파형" />
              <WaveformPreview csvUrl={item.timeseries_url ?? null} title="유사 사례 파형" />
            </div>
          </div>
          <div className="modalCaseFacts">
            <div className="caseTitle">
              <span className="status completed">{retrieverModeLabel(item)}</span>
              <strong>{item.sample_id}</strong>
            </div>
            {props.status === "loading" ? (
              <div className="modalLoading"><Loader2 className="spin" size={16} /> 상세 정보 동기화 중</div>
            ) : null}
            {props.status === "failed" ? <p className="ragError">상세 API를 불러오지 못했습니다. 현재 trace의 사례 정보로 표시합니다.</p> : null}
            <dl>
              <dt>유사도</dt><dd>{formatMetric(item.similarity)}</dd>
              <dt>방전유형</dt><dd>{item.label_name}</dd>
              <dt>파형 피크</dt><dd>{metadataValue(item, "max_discharge_value")}</dd>
            </dl>
            <CaseComparisonTable rows={comparisonRows} />
            <CaseComparisonPoints points={comparisonPoints} />
            <SimilarityComponentBars item={item} />
            <div className="reasonBox">
              <strong>매칭 근거</strong>
              <p>{item.reason}</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function CaseComparisonTable(props: { readonly rows: readonly CaseComparisonRow[] }) {
  return (
    <div className="caseComparisonTable">
      <strong>현재 점검 비교표</strong>
      <div className="caseComparisonHeader">
        <span>항목</span>
        <span>현재 점검</span>
        <span>유사 사례</span>
      </div>
      {props.rows.map((row) => (
        <div className={`caseComparisonRow ${row.match}`} key={row.label}>
          <span>{row.label}</span>
          <strong>{row.currentValue}</strong>
          <strong>{row.similarValue}</strong>
        </div>
      ))}
    </div>
  );
}

function CaseComparisonPoints(props: { readonly points: readonly CaseComparisonPoint[] }) {
  if (props.points.length === 0) {
    return null;
  }
  return (
    <div className="caseComparisonPoints">
      {props.points.map((point) => (
        <div className={`caseComparisonPoint ${point.tone}`} key={`${point.tone}-${point.title}`}>
          <span>{point.tone === "different" ? "상이" : point.tone === "same" ? "유사" : "참고"}</span>
          <strong>{point.title}</strong>
          <p>{point.detail}</p>
        </div>
      ))}
    </div>
  );
}

function EvidenceImagePanel(props: { readonly imageUrl: string | null; readonly title: string }) {
  return (
    <div className="modalImageFrame">
      <span>{props.title}</span>
      {props.imageUrl === null ? (
        <div className="previewEmpty">PRPD 이미지 없음</div>
      ) : (
        <img alt={props.title} src={apiAssetUrl(props.imageUrl)} />
      )}
    </div>
  );
}

function WaveformPreview(props: { readonly csvUrl: string | null; readonly title: string }) {
  const [status, setStatus] = useState<"idle" | "loading" | "failed">("idle");
  const [points, setPoints] = useState<readonly number[]>([]);

  useEffect(() => {
    if (props.csvUrl === null) {
      setStatus("idle");
      setPoints([]);
      return;
    }
    let cancelled = false;
    setStatus("loading");
    fetch(apiAssetUrl(props.csvUrl))
      .then((response) => {
        if (!response.ok) {
          throw new Error(`waveform request failed: ${response.status}`);
        }
        return response.text();
      })
      .then((text) => {
        if (!cancelled) {
          setPoints(parseWaveformCsv(text));
          setStatus("idle");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPoints([]);
          setStatus("failed");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [props.csvUrl]);

  const path = waveformPolyline(points, 220, 82);
  const statusText = waveformStatusText(props.csvUrl, status, points.length);
  return (
    <div className="waveformPreview">
      <div>
        <span>{props.title}</span>
        <strong>{statusText}</strong>
      </div>
      <svg viewBox="0 0 220 82" role="img" aria-label={`${props.title} 미리보기`}>
        <line x1="0" y1="41" x2="220" y2="41" />
        {path === "" ? null : <polyline points={path} />}
      </svg>
    </div>
  );
}

function SimilarityComponentBars(props: { readonly compact?: boolean; readonly item: SimilarCase }) {
  const items = similarityComponentItems(props.item);
  if (items.length === 0) {
    return null;
  }
  return (
    <div className={`similarityComponents ${props.compact === true ? "compact" : ""}`}>
      {props.compact === true ? null : <strong>유사도 구성</strong>}
      {items.map((item) => (
        <div className="similarityComponentRow" key={item.label}>
          <span>{item.label}</span>
          <div aria-label={`${item.label} ${formatMetric(item.score)}`}>
            <b style={{width: `${similarityWidth(item.score)}%`}} />
          </div>
          <em>{formatMetric(item.score)}</em>
        </div>
      ))}
    </div>
  );
}

function similarityComponentItems(item: SimilarCase): readonly SimilarityComponentItem[] {
  const featureComponents = [
    {label: "PRPD", score: metadataNumber(item, "feature_component_prpd")},
    {label: "시계열", score: metadataNumber(item, "feature_component_timeseries")},
  ].filter((component): component is SimilarityComponentItem => component.score !== null);
  if (featureComponents.length > 0) {
    return featureComponents;
  }
  const embeddingSimilarity = metadataNumber(item, "embedding_similarity");
  return embeddingSimilarity === null ? [] : [{label: "Embedding", score: embeddingSimilarity}];
}

function similarCaseRankSummary(item: SimilarCase, index: number): SimilarCaseRankSummary {
  const prpdScore = metadataNumber(item, "feature_component_prpd");
  const timeseriesScore = metadataNumber(item, "feature_component_timeseries");
  const prefix = `#${index + 1}`;
  return {
    title: `${prefix} ${modalityRankReason(prpdScore, timeseriesScore)}`,
    detail: modalityScoreSummary(prpdScore, timeseriesScore),
  };
}

function modalityRankReason(prpdScore: number | null, timeseriesScore: number | null): string {
  if (prpdScore === null && timeseriesScore === null) {
    return "PRPD/파형 feature 대조";
  }
  if (prpdScore === null) {
    return "파형 기준 유사";
  }
  if (timeseriesScore === null) {
    return "PRPD 기준 유사";
  }
  if (prpdScore >= 0.72 && timeseriesScore >= 0.72) {
    return "PRPD/파형 모두 유사";
  }
  if (prpdScore - timeseriesScore >= 0.15) {
    return "PRPD 중심 유사";
  }
  if (timeseriesScore - prpdScore >= 0.15) {
    return "파형 중심 유사";
  }
  return "PRPD/파형 균형 유사";
}

function modalityScoreSummary(prpdScore: number | null, timeseriesScore: number | null): string {
  return `PRPD ${modalityScoreLabel(prpdScore)} · 파형 ${modalityScoreLabel(timeseriesScore)}`;
}

function modalityScoreLabel(score: number | null): string {
  if (score === null) {
    return "없음";
  }
  if (score >= 0.72) {
    return "유사";
  }
  if (score >= 0.45) {
    return "보통";
  }
  return "상대 약함";
}

function parseWaveformCsv(text: string): readonly number[] {
  const values: number[] = [];
  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    if (values.length >= 2400) {
      break;
    }
    const fields = line.trim().split(/[,\t; ]+/).map((field) => Number(field)).filter(Number.isFinite);
    const firstValue = fields[0];
    if (firstValue !== undefined) {
      values.push(firstValue);
    }
  }
  return downsampleWaveform(values, 120);
}

function downsampleWaveform(values: readonly number[], targetCount: number): readonly number[] {
  if (values.length <= targetCount) {
    return values;
  }
  const result: number[] = [];
  const bucketSize = values.length / targetCount;
  for (let index = 0; index < targetCount; index += 1) {
    const start = Math.floor(index * bucketSize);
    const end = Math.max(start + 1, Math.floor((index + 1) * bucketSize));
    const bucket = values.slice(start, end);
    result.push(bucket.reduce((sum, value) => sum + value, 0) / bucket.length);
  }
  return result;
}

function waveformPolyline(points: readonly number[], width: number, height: number): string {
  if (points.length < 2) {
    return "";
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  return points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function waveformStatusText(csvUrl: string | null, status: "idle" | "loading" | "failed", pointCount: number): string {
  if (csvUrl === null) {
    return "CSV 없음";
  }
  if (status === "loading") {
    return "불러오는 중";
  }
  if (status === "failed") {
    return "불러오기 실패";
  }
  return pointCount === 0 ? "파형 없음" : `${pointCount} pts`;
}

function caseComparisonRows(
  item: SimilarCase,
  trace: TraceResponse | null,
  currentArtifacts: InputArtifactEvidence | null,
): readonly CaseComparisonRow[] {
  const fusion = trace === null ? null : parseFusionSummary(findTraceEvent(trace, "fusion_engine")?.summary);
  const finalLabel = fusion?.final_label_name ?? currentModelLabel(trace);
  return [
    textComparisonRow("방전유형", finalLabel, item.label_name),
    textComparisonRow("PRPD 이미지", artifactPresenceText(currentArtifacts?.prpdImageUrl), artifactPresenceText(item.image_url)),
    textComparisonRow("시계열 CSV", artifactPresenceText(currentArtifacts?.timeseriesCsvUrl), artifactPresenceText(item.timeseries_url ?? null)),
    numericComparisonRow("신호 피크", currentArtifacts?.signalSummary?.peak_abs ?? null, metadataNumber(item, "max_discharge_value")),
    {
      label: "유사도",
      currentValue: "기준 사례",
      similarValue: formatMetric(item.similarity),
      match: item.similarity >= 0.72 ? "same" : item.similarity >= 0.45 ? "unknown" : "different",
    },
  ];
}

function caseComparisonPoints(
  item: SimilarCase,
  rows: readonly CaseComparisonRow[],
): readonly CaseComparisonPoint[] {
  const points: CaseComparisonPoint[] = [];
  const componentItems = similarityComponentItems(item);
  const strongComponents = componentItems.filter((component) => component.score >= 0.65);
  if (strongComponents.length > 0) {
    points.push({
      tone: "same",
      title: `${strongComponents.map((component) => component.label).join(", ")} 유사도 높음`,
      detail: "계산된 유사도 구성 점수에서 강하게 가까운 축입니다.",
    });
  }
  const sameRows = rows.filter((row) => row.match === "same" && row.label !== "유사도");
  if (sameRows.length > 0) {
    points.push({
      tone: "same",
      title: `${sameRows.slice(0, 3).map((row) => row.label).join(", ")} 일치`,
      detail: "현재 점검과 과거 사례에서 직접 비교 가능한 핵심 신호 항목입니다.",
    });
  }
  if (item.similarity >= 0.72) {
    points.push({
      tone: "same",
      title: `전체 유사도 ${formatMetric(item.similarity)}`,
      detail: "현재 점검의 자동 추천 순위에서 상위 후보로 볼 수 있는 수준입니다.",
    });
  }
  const differentRows = rows.filter((row) => row.match === "different" && row.label !== "유사도");
  if (differentRows.length > 0) {
    points.push({
      tone: "different",
      title: `${differentRows.slice(0, 3).map((row) => row.label).join(", ")} 차이`,
      detail: "유사하지만 완전히 같은 조건의 사례는 아니므로 해석 시 같이 확인해야 합니다.",
    });
  }
  if (points.length === 0) {
    return [{
      tone: "neutral",
      title: "비교 가능한 근거 제한",
      detail: "현재 trace 또는 과거 사례에 일부 값이 없어 원본 PRPD/파형 중심으로 확인해야 합니다.",
    }];
  }
  return points.slice(0, 4);
}

function textComparisonRow(label: string, currentValue: string | null | undefined, similarValue: string | null | undefined): CaseComparisonRow {
  const currentText = displayComparisonValue(currentValue);
  const similarText = displayComparisonValue(similarValue);
  return {
    label,
    currentValue: currentText,
    similarValue: similarText,
    match: comparisonMatch(currentText, similarText),
  };
}

function numericComparisonRow(label: string, currentValue: number | null, similarValue: number | null): CaseComparisonRow {
  return {
    label,
    currentValue: formatNumeric(currentValue),
    similarValue: formatNumeric(similarValue),
    match: numericComparisonMatch(currentValue, similarValue),
  };
}

function currentModelLabel(trace: TraceResponse | null): string {
  if (trace === null) {
    return "없음";
  }
  const timeSeries = summaryValue(findTraceEvent(trace, "time_series_tool"), "label_name");
  const vision = summaryValue(findTraceEvent(trace, "vision_tool"), "label_name");
  if (timeSeries !== "없음" && timeSeries === vision) {
    return timeSeries;
  }
  const fusionLabel = summaryValue(findTraceEvent(trace, "fusion_engine"), "final_label_name");
  return fusionLabel === "없음" ? firstMeaningfulText(timeSeries, vision) : fusionLabel;
}

function artifactPresenceText(value: string | null | undefined): string {
  return value === undefined || value === null || value.trim().length === 0 ? "없음" : "있음";
}

function displayComparisonValue(value: string | null | undefined): string {
  const text = value?.trim();
  return text === undefined || text.length === 0 || isMissingComparisonValue(text) ? "없음" : text;
}

function comparisonMatch(currentValue: string, similarValue: string): CaseComparisonRow["match"] {
  if (isMissingComparisonValue(currentValue) || isMissingComparisonValue(similarValue)) {
    return "unknown";
  }
  return normalizeComparisonValue(currentValue) === normalizeComparisonValue(similarValue) ? "same" : "different";
}

function numericComparisonMatch(currentValue: number | null, similarValue: number | null): CaseComparisonRow["match"] {
  if (currentValue === null || similarValue === null || !Number.isFinite(currentValue) || !Number.isFinite(similarValue)) {
    return "unknown";
  }
  const denominator = Math.max(Math.abs(currentValue), Math.abs(similarValue), 1);
  return Math.abs(currentValue - similarValue) / denominator <= 0.25 ? "same" : "different";
}

function normalizeComparisonValue(value: string): string {
  return value
    .toLowerCase()
    .replace(/[\s"'[\](),]/g, "")
    .replace(/millimeters?/g, "mm");
}

function isMissingComparisonValue(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return ["", "없음", "n/a", "na", "null", "undefined", "unknown", "미정"].includes(normalized);
}

function firstMeaningfulText(...values: readonly string[]): string {
  return values.find((value) => !isMissingComparisonValue(value)) ?? "없음";
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

function rankedReportRagDocuments(documents: readonly RagDocument[]): readonly ReportRagDocument[] {
  return [...documents]
    .sort((left, right) => right.relevance - left.relevance)
    .slice(0, REPORT_RAG_DOCUMENT_LIMIT)
    .map((document, index) => {
      const facts = ragDocumentFacts(document);
      return {
        document,
        facts: facts.slice(0, REPORT_RAG_FACT_LIMIT),
        hiddenFactCount: Math.max(0, facts.length - REPORT_RAG_FACT_LIMIT),
        rank: index + 1,
        retrievalLabel: retrievalModeLabel(ragRetrievalMode(document)),
        scoreTone: ragScoreTone(document.relevance),
        sourceLabel: sourceTypeLabel(document.source_type ?? undefined),
        summary: ragDocumentSummary(document, facts),
      };
    });
}

function reportRagEvidenceSummary(
  documents: readonly RagDocument[],
  shownCount: number,
): RagEvidenceSummary {
  return {
    documentCount: documents.length,
    retrievalText: distributionText(documents.map((document) => retrievalModeLabel(ragRetrievalMode(document))), "없음"),
    shownCount,
    sourceText: distributionText(documents.map((document) => sourceTypeLabel(document.source_type ?? undefined)), "없음"),
    topScore: topDocumentScore(documents),
  };
}

function distributionText(values: readonly string[], emptyText: string): string {
  if (values.length === 0) {
    return emptyText;
  }
  const counts = new Map<string, number>();
  values.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 2)
    .map(([label, count]) => `${label} ${count}`)
    .join(" · ");
}

function evidenceLinksForSignal(
  signal: ModelSignal,
  finalLabel: string,
  documents: readonly RagDocument[],
  cases: readonly SimilarCase[],
): readonly ReportEvidenceLink[] {
  const targetLabel = signal.label === "없음" || signal.label === "근거 검색" ? finalLabel : signal.label;
  const matchingDocuments = rankedDocumentsForLabel(documents, targetLabel);
  const matchingCases = rankedCasesForLabel(cases, targetLabel);
  if (signal.source === "RAG") {
    return matchingDocuments.slice(0, 3);
  }
  if (signal.source === "유사 사례") {
    return matchingCases.slice(0, 3);
  }
  return [...matchingDocuments.slice(0, 2), ...matchingCases.slice(0, 1)];
}

function reportContributionItems(
  documents: readonly RagDocument[],
  cases: readonly SimilarCase[],
  fusion: FusionSummaryPayload | null,
): readonly ContributionItem[] {
  const ragScore = topDocumentScore(documents);
  const caseScore = topCaseScore(cases);
  return [
    {
      label: "RAG 문서",
      score: ragScore,
      summary: fusionSourceText("rag", fusion, documents.length),
      details: topDocumentDetails(documents),
    },
    {
      label: "유사 사례",
      score: caseScore,
      summary: fusionSourceText("similar_case", fusion, cases.length),
      details: topCaseDetails(cases),
    },
  ];
}

function rankedDocumentsForLabel(documents: readonly RagDocument[], label: string): readonly ReportEvidenceLink[] {
  return documents
    .map((document) => ({
      kind: "document" as const,
      reason: documentReason(document, label),
      score: document.relevance,
      title: document.title,
    }))
    .sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
}

function rankedCasesForLabel(cases: readonly SimilarCase[], label: string): readonly ReportEvidenceLink[] {
  return cases
    .map((item) => ({
      kind: "case" as const,
      reason: item.label_name === label ? "판정 라벨 일치" : `${item.label_name} 참조`,
      score: item.similarity,
      title: item.sample_id,
    }))
    .sort((left, right) => (right.score ?? 0) - (left.score ?? 0));
}

function documentReason(document: RagDocument, label: string): string {
  const documentLabel = stringValue(document.metadata["label_name"]);
  if (documentLabel === label) {
    return "문서 라벨 일치";
  }
  if (document.source_type === "rulebook") {
    return "규칙서 기준";
  }
  return documentLabel === undefined ? "검색 관련도" : `${documentLabel} 근거`;
}

function fusionSourceText(source: string, fusion: FusionSummaryPayload | null, count: number): string {
  if (count === 0) {
    return "최종 판정에 연결된 근거가 없습니다.";
  }
  const included = fusion?.contributing_sources.includes(source) ?? false;
  return included ? "융합 판단의 반영 근거로 사용되었습니다." : "융합 판단 전후 검증 근거로 참고되었습니다.";
}

function topDocumentDetails(documents: readonly RagDocument[]): readonly string[] {
  if (documents.length === 0) {
    return ["검색된 RAG 문서 없음"];
  }
  return documents.slice(0, 3).map((document) => `${document.title} · 관련도 ${formatMetric(document.relevance)}`);
}

function topCaseDetails(cases: readonly SimilarCase[]): readonly string[] {
  if (cases.length === 0) {
    return ["매칭된 유사 사례 없음"];
  }
  return cases.slice(0, 3).map((item) => `${item.sample_id} · ${item.label_name} · 유사도 ${formatMetric(item.similarity)}`);
}

function topDocumentScore(documents: readonly RagDocument[]): number | null {
  if (documents.length === 0) {
    return null;
  }
  return Math.max(...documents.map((document) => document.relevance));
}

function topCaseScore(cases: readonly SimilarCase[]): number | null {
  if (cases.length === 0) {
    return null;
  }
  return Math.max(...cases.map((item) => item.similarity));
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

function llmRagStateLabel(status: ModelRuntimeStatus | null): string {
  if (status === null) {
    return "상태 확인 중";
  }
  if (status.llm_rag_ready) {
    return "OpenRouter 활성";
  }
  if (status.llm_rag_provider === "mock") {
    return "Mock provider";
  }
  return "Fallback";
}

function llmRagStateDetail(status: ModelRuntimeStatus | null): string {
  if (status === null) {
    return "백엔드 런타임 상태를 불러오고 있습니다.";
  }
  if (status.llm_rag_ready) {
    return `${status.llm_rag_model ?? "OpenRouter 모델"}은 RAG 챗과 보조 설명에 사용됩니다. 진단 VLM은 별도 어댑터를 사용합니다.`;
  }
  if (status.llm_rag_error !== null) {
    return status.llm_rag_error;
  }
  return `${status.llm_rag_adapter} 상태입니다. 진단 VLM 어댑터는 별도로 동작합니다.`;
}

function llmRagFlowSteps(status: ModelRuntimeStatus | null): readonly LlmRagFlowStep[] {
  const active = status?.llm_rag_ready ?? false;
  const vlmReady = status?.vlm_ready ?? false;
  return [
    {
      detail: status === null ? "상태 확인 중" : `${status.rag_retriever} ${status.rag_version}`,
      label: "retrieval",
      state: status === null ? "neutral" : "ready",
      title: "RAG 문서 검색",
    },
    {
      detail: "문서, 유사 사례, 시계열/비전 결과를 보조 설명 입력으로 구성",
      label: "prompt",
      state: "ready",
      title: "근거 프롬프트",
    },
    {
      detail: active ? `${status?.llm_rag_model ?? "OpenRouter"} JSON 응답 사용` : "OpenRouter 비활성 또는 설정 미완료",
      label: "llm",
      state: active ? "ready" : "warn",
      title: "OpenRouter 보조 LLM",
    },
    {
      detail: vlmReady ? `${status?.vlm_model ?? "진단 VLM"} 진단 어댑터 준비됨` : "진단 VLM 상태 확인 필요",
      label: "diagnosis-vlm",
      state: vlmReady ? "ready" : "warn",
      title: "진단 VLM",
    },
  ];
}

function sourceTypeLabel(sourceType: string | undefined): string {
  const labels: Record<string, string> = {
    dataset_case: "과거 사례",
    rulebook: "판정 기준",
    sop: "운영 절차",
  };
  return sourceType === undefined ? "없음" : labels[sourceType] ?? sourceType;
}

function ragRetrievalMode(document: RagDocument): string {
  return document.retrieval_mode ?? stringValue(document.metadata["retrieval_mode"]) ?? "semantic_similarity";
}

function retrievalModeLabel(mode: string): string {
  const labels: Record<string, string> = {
    deterministic_fallback: "Fallback",
    exact_sample_id: "정확 샘플ID",
    metadata_filter: "조건 일치",
    rulebook_semantic: "기준 문서",
    semantic_similarity: "유사도 검색",
    sop_semantic: "SOP",
  };
  return labels[mode] ?? mode;
}

function ragDocumentTitle(document: RagDocument): string {
  return document.title.replace(/^데이터셋 사례\s+/, "");
}

function documentKeyFromRagDocument(document: RagDocument): string {
  const [documentKey] = document.document_id.split("#", 1);
  return documentKey || document.document_id;
}

function ragDetailTitle(detail: RagDocumentDetailResponse): string {
  return detail.title.replace(/^데이터셋 사례\s+/, "");
}

function ragDetailLead(detail: RagDocumentDetailResponse): string {
  if (detail.source_type !== "dataset_case") {
    return detail.source_path ?? detail.document_key;
  }
  const label = metadataText(detail.metadata, "label_name");
  const equipment = metadataText(detail.metadata, "equipment_name");
  const sensor = metadataText(detail.metadata, "sensor_type");
  return [label, equipment, sensor].filter((value) => value !== "없음").join(" · ") || detail.document_key;
}

function ragDocumentDetailFacts(detail: RagDocumentDetailResponse): readonly RagFact[] {
  const metadata = detail.metadata;
  return [
    {label: "방전유형", value: metadataText(metadata, "label_name")},
    {label: "설비", value: metadataText(metadata, "equipment_name")},
    {label: "센서", value: metadataText(metadata, "sensor_type")},
    {label: "절연", value: metadataText(metadata, "insulator_type", "insulator_name")},
    {label: "전압", value: metadataText(metadata, "equipment_rated_voltage")},
    {label: "이격", value: metadataText(metadata, "clearance_distance")},
    {label: "피크", value: metadataText(metadata, "max_discharge_value")},
    {label: "출처", value: detail.source_path ?? detail.document_key},
  ].filter((fact) => fact.value !== "없음");
}

function ragDocumentFacts(document: RagDocument): readonly RagFact[] {
  const metadata = document.metadata;
  return [
    {label: "방전", value: metadataText(metadata, "label_name")},
    {label: "설비", value: metadataText(metadata, "equipment_name")},
    {label: "센서", value: metadataText(metadata, "sensor_type")},
    {label: "절연", value: metadataText(metadata, "insulator_type", "insulator_name")},
    {label: "전압", value: metadataText(metadata, "equipment_rated_voltage")},
    {label: "전류", value: metadataText(metadata, "equipment_rated_current")},
    {label: "이격", value: metadataText(metadata, "clearance_distance")},
    {label: "온습도", value: temperatureHumidityText(metadata)},
    {label: "피크", value: metadataText(metadata, "max_discharge_value")},
  ].filter((fact) => fact.value !== "없음");
}

function ragDocumentSummary(document: RagDocument, facts: readonly RagFact[]): string {
  if (document.source_type !== "dataset_case") {
    return compactText(document.excerpt, 180);
  }
  const patternSummary = metadataText(document.metadata, "pattern_summary");
  if (patternSummary !== "없음") {
    return patternSummary;
  }
  if (facts.length > 0) {
    return `${ragDocumentTitle(document)} 사례는 ${facts.slice(0, 4).map((fact) => `${fact.label} ${fact.value}`).join(", ")} 조건의 과거 진단 근거입니다.`;
  }
  return compactText(cleanRagExcerpt(document.excerpt), 180);
}

function metadataText(metadata: Record<string, string | number | null>, ...keys: readonly string[]): string {
  for (const key of keys) {
    const value = stringValue(metadata[key]);
    if (value !== undefined && value.toLowerCase() !== "nan") {
      return value;
    }
  }
  return "없음";
}

function temperatureHumidityText(metadata: Record<string, string | number | null>): string {
  const temperature = metadataText(metadata, "temperature");
  const humidity = metadataText(metadata, "humidity");
  if (temperature === "없음" && humidity === "없음") {
    return "없음";
  }
  if (temperature === "없음") {
    return `${humidity}%`;
  }
  if (humidity === "없음") {
    return `${temperature}°C`;
  }
  return `${temperature}°C / ${humidity}%`;
}

function cleanRagExcerpt(excerpt: string): string {
  return excerpt
    .replace(/\b(sample_id|label_id|label|equipment|manufacturer|equipment_type|equipment_id|sensor|insulator|insulator_name|voltage|current|clearance|recording_time|duration|power_supply|power_frequency|temperature|humidity|iec_standard|defect_nums|defect_details|max_discharge)=/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function ragReadableLabel(key: string): string {
  const labels: Record<string, string> = {
    clearance: "이격",
    current: "정격전류",
    defect_details: "결함상세",
    defect_nums: "결함수",
    duration: "기록길이",
    equipment: "설비",
    equipment_id: "설비ID",
    equipment_type: "설비유형",
    humidity: "습도",
    iec_standard: "IEC",
    insulator: "절연",
    insulator_name: "절연명",
    label: "방전유형",
    label_id: "라벨ID",
    manufacturer: "제조사",
    max_discharge: "최대방전",
    pattern_summary: "요약",
    power_frequency: "전원주파수",
    power_supply: "전원",
    recording_time: "기록시각",
    sample_id: "샘플ID",
    sensor: "센서",
    temperature: "온도",
    voltage: "정격전압",
  };
  return labels[key] ?? key;
}

function ragScoreTone(score: number): string {
  if (score >= 0.35) {
    return "strong";
  }
  if (score >= 0.15) {
    return "medium";
  }
  return "weak";
}

function queryMetadataSummary(metadata: Record<string, unknown>): string {
  const topK = metadata["top_k"];
  const sourceTypes = metadata["source_types"];
  const topKText = typeof topK === "number" ? `topK ${topK}` : "topK 없음";
  const sourceText = Array.isArray(sourceTypes) ? sourceTypes.map(String).join(", ") : "source 전체";
  return `${topKText} · ${sourceText}`;
}

function compactText(value: string, maxLength: number): string {
  const clean = value.split(/\s+/).join(" ").trim();
  if (clean.length <= maxLength) {
    return clean;
  }
  return `${clean.slice(0, maxLength)}...`;
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

function ragMetadataMissingSummary(status: RagStatusResponse | null): { readonly text: string; readonly tone: string } {
  if (status === null) {
    return {text: "확인 중", tone: "neutral"};
  }
  const missing = Object.entries(status.metadata_missing_counts)
    .filter(([, count]) => count > 0)
    .sort(([, left], [, right]) => right - left);
  if (missing.length === 0) {
    return {text: "없음", tone: "ready"};
  }
  return {
    text: missing.map(([key, count]) => `${ragMetadataFieldLabel(key)} ${count}`).join(", "),
    tone: "warn",
  };
}

function ragMetadataFieldLabel(key: string): string {
  const labels: Record<string, string> = {
    humidity: "습도",
    label_name: "라벨",
    max_discharge_value: "피크",
    recording_time: "기록시각",
    sample_id: "샘플",
    temperature: "온도",
  };
  return labels[key] ?? key;
}

function boundedRagTopK(value: number): number {
  if (!Number.isFinite(value)) {
    return RAG_TOP_K_MIN;
  }
  return Math.min(RAG_TOP_K_MAX, Math.max(RAG_TOP_K_MIN, Math.trunc(value)));
}

function boundedSimilarCaseLimit(value: number, max: number): number {
  const upperBound = Math.max(1, Math.trunc(max));
  if (!Number.isFinite(value)) {
    return Math.min(DEFAULT_SIMILAR_CASE_LIMIT, upperBound);
  }
  return Math.min(upperBound, Math.max(1, Math.trunc(value)));
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

function metricWidth(value: number | null): number {
  if (value === null) {
    return 0;
  }
  return Math.max(2, Math.min(100, Math.round(value * 100)));
}

function metadataValue(item: SimilarCase, key: string): string {
  const value = item.metadata[key];
  if (value === undefined || value === null || value === "") {
    return "없음";
  }
  return String(value);
}

function metadataNumber(item: SimilarCase, key: string): number | null {
  const value = item.metadata[key];
  const numericValue = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function retrieverModeLabel(item: SimilarCase): string {
  const mode = metadataValue(item, "retriever_mode");
  if (mode === "learned_projection_encoder") {
    return "학습형 파형 embedding";
  }
  if (mode === "domain_feature_retriever") {
    return "도메인 feature";
  }
  if (mode === "없음") {
    return "도메인 feature";
  }
  return mode;
}

async function loadDashboard(): Promise<{
  backendStatus: BackendStatus;
  history: readonly DiagnosisListItem[];
  modelRuntime: ModelRuntimeStatus;
  ragStatus: RagStatusResponse;
  reviewQueue: readonly DiagnosisListItem[];
}> {
  const [health, history, modelRuntime, ragStatus, reviewQueue] = await Promise.all([
    fetchHealth(),
    fetchDiagnosisHistory(),
    fetchModelRuntimeStatus(),
    fetchRagStatus(),
    fetchReviewQueue(),
  ]);
  return {
    backendStatus: health.status === "ok" ? "online" : "offline",
    history: history.items,
    modelRuntime,
    ragStatus,
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
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
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
