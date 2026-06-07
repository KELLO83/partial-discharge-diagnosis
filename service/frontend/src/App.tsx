import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpenText,
  Cpu,
  Database,
  FileImage,
  FileText,
  Gauge,
  Loader2,
  RefreshCw,
  Send,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  fetchDiagnosisHistory,
  fetchDiagnosisTrace,
  fetchHealth,
  fetchModelStatus,
  fetchReviewQueue,
  submitDiagnosis,
} from "./api";
import { fileDisplayText } from "./fileDisplay";
import { buildInputPresence } from "./formState";
import { DEFAULT_METADATA } from "./metadataDefaults";
import { selectInputRoute } from "./route";
import type {
  DiagnosisListItem,
  DiagnosisResponse,
  DiagnosisStatus,
  MetadataForm,
  ModelRuntimeStatus,
  TraceResponse,
} from "./types";
import "./styles.css";

type RequestStatus = "idle" | "submitting" | "failed";
type BackendStatus = "checking" | "online" | "offline";

const routeLabel = {
  hybrid: "Hybrid",
  timeseries_only: "Time-series",
  vlm_only: "VLM",
  insufficient_input: "Input pending",
};

const statusLabel: Record<DiagnosisStatus, string> = {
  completed: "complete",
  needs_review: "review",
  rejected: "rejected",
};

export function App() {
  const [image, setImage] = useState<File | null>(null);
  const [csv, setCsv] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<MetadataForm>(DEFAULT_METADATA);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [history, setHistory] = useState<readonly DiagnosisListItem[]>([]);
  const [reviewQueue, setReviewQueue] = useState<readonly DiagnosisListItem[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelRuntimeStatus | null>(null);
  const [selectedHistory, setSelectedHistory] = useState<DiagnosisListItem | null>(null);
  const [expandedTraceKey, setExpandedTraceKey] = useState<string | null>(null);
  const [requestStatus, setRequestStatus] = useState<RequestStatus>("idle");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const route = useMemo(
    () => selectInputRoute(buildInputPresence({ hasImage: image !== null, hasTimeseries: csv !== null, metadata })),
    [image, csv, metadata],
  );

  useEffect(() => {
    let active = true;
    void loadDashboard()
      .then((dashboard) => {
        if (active) {
          setBackendStatus(dashboard.backendStatus);
          setHistory(dashboard.history);
          setReviewQueue(dashboard.reviewQueue);
          setModelStatus(dashboard.modelStatus);
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

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setRequestStatus("submitting");
    setTrace(null);
    try {
      const response = await submitDiagnosis({ image, csv, metadata });
      setResult(response);
      setSelectedHistory(null);
      setTrace(await fetchDiagnosisTrace(response.diagnosis_id));
      await refreshLists().catch(() => undefined);
      setRequestStatus("idle");
    } catch {
      setRequestStatus("failed");
    }
  }

  async function refreshLists(): Promise<void> {
    const [historyResponse, queueResponse, statusResponse] = await Promise.all([
      fetchDiagnosisHistory(),
      fetchReviewQueue(),
      fetchModelStatus(),
    ]);
    setHistory(historyResponse.items);
    setReviewQueue(queueResponse.items);
    setModelStatus(statusResponse);
  }

  async function openDiagnosis(item: DiagnosisListItem): Promise<void> {
    setSelectedHistory(item);
    setResult(null);
    setExpandedTraceKey(null);
    setTrace(await fetchDiagnosisTrace(item.diagnosis_id));
  }

  return (
    <main className="shell">
      <aside className="sidebar" aria-label="Primary">
        <div className="brandBlock">
          <div className="brandMark">PD</div>
          <div>
            <p className="eyebrow">Industrial</p>
            <h1>Diagnosis Agent</h1>
          </div>
        </div>
        <nav className="navList">
          <a href="#overview"><Gauge size={18} />Overview</a>
          <a href="#intake"><Database size={18} />Inspection</a>
          <a href="#verdict"><ShieldCheck size={18} />Verdict</a>
          <a href="#history"><Database size={18} />History</a>
          <a href="#queue"><AlertTriangle size={18} />Queue</a>
          <a href="#trace"><Cpu size={18} />Trace</a>
          <a href="#wiki"><BookOpenText size={18} />Wiki</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Inspection Console</p>
            <h2>Partial Discharge Composite Diagnosis</h2>
          </div>
          <button className="iconButton" type="button" onClick={() => window.location.reload()}>
            <RefreshCw size={18} />
            Refresh
          </button>
        </header>

        <section id="overview" className="kpiGrid">
          <MetricCard icon={<Database size={20} />} label="Backend" value={backendStatusLabel(backendStatus)} tone={backendTone(backendStatus)} />
          <MetricCard icon={<Activity size={20} />} label="Route" value={routeLabel[route]} tone="blue" />
          <MetricCard icon={<BarChart3 size={20} />} label="History" value={history.length.toString()} tone="amber" />
          <MetricCard icon={<Cpu size={20} />} label="Trace Steps" value={(trace?.events.length ?? 0).toString()} tone="violet" />
        </section>

        <section className="twoColumn">
          <Panel id="intake" title="Inspection Intake" action={routeLabel[route]}>
            <form className="intakeForm" onSubmit={(event) => void handleSubmit(event)}>
              <div className="uploadGrid">
                <UploadField
                  accept="image/png"
                  description={fileDisplayText(image, "PRPD PNG")}
                  icon={<FileImage size={20} />}
                  label="PRPD Image"
                  onChange={setImage}
                />
                <UploadField
                  accept=".csv,text/csv"
                  description={fileDisplayText(csv, "time-series CSV")}
                  icon={<FileText size={20} />}
                  label="Time-series CSV"
                  onChange={setCsv}
                />
              </div>

              <div className="formSection">
                <div className="sectionTitle">Equipment metadata</div>
                <div className="fieldGrid">
                  <Field label="Equipment" value={metadata.equipmentName} onChange={(value) => setMetadata({ ...metadata, equipmentName: value })} />
                  <Field label="Rated voltage" value={metadata.ratedVoltage} onChange={(value) => setMetadata({ ...metadata, ratedVoltage: value })} />
                  <Field label="Rated current" value={metadata.ratedCurrent} onChange={(value) => setMetadata({ ...metadata, ratedCurrent: value })} />
                  <Field label="Sensor" value={metadata.sensorType} onChange={(value) => setMetadata({ ...metadata, sensorType: value })} />
                  <Field label="Temperature" value={metadata.temperature} onChange={(value) => setMetadata({ ...metadata, temperature: value })} />
                  <Field label="Humidity" value={metadata.humidity} onChange={(value) => setMetadata({ ...metadata, humidity: value })} />
                </div>
              </div>

              <div className="formFooter">
                <div>
                  <span>Selected workflow</span>
                  <strong>{routeLabel[route]}</strong>
                </div>
                <button className="primaryButton" type="submit" disabled={requestStatus === "submitting"}>
                  {requestStatus === "submitting" ? <Loader2 className="spin" size={17} /> : <Send size={17} />}
                  {requestStatus === "submitting" ? "Running" : "Run diagnosis"}
                </button>
              </div>
            </form>
          </Panel>

          <Panel id="verdict" title="Agent Verdict" action={result === null ? "pending" : statusLabel[result.status]}>
            <VerdictPanel historyItem={selectedHistory} result={result} requestStatus={requestStatus} />
          </Panel>
        </section>

        <section className="twoColumn">
          <Panel id="history" title="Diagnosis History" action={`${history.length} records`}>
            <DiagnosisTable items={history} onOpen={(item) => void openDiagnosis(item)} />
          </Panel>

          <Panel id="queue" title="Review Queue" action={`${reviewQueue.length} items`}>
            <ReviewQueue items={reviewQueue} onOpen={(item) => void openDiagnosis(item)} />
          </Panel>
        </section>

        <section id="trace" className="panel">
          <div className="panelHeader">
            <h3>Trace Log</h3>
            <span>{trace?.trace_id ?? "no trace"}</span>
          </div>
          <TraceLog expandedKey={expandedTraceKey} onToggle={setExpandedTraceKey} trace={trace} />
        </section>

        <section id="runtime" className="panel">
          <div className="panelHeader">
            <h3>Model Runtime Status</h3>
            <span>{modelStatus?.agent_mode ?? "pending"}</span>
          </div>
          <ModelStatusPanel modelStatus={modelStatus} />
        </section>

        <section id="wiki" className="panel">
          <div className="panelHeader">
            <h3>Model Integration Notes</h3>
            <span>adapter contract</span>
          </div>
          <div className="noteGrid">
            <NoteItem title="Time-series tool" body="TimeSeriesInferenceAdapter.run(input) must return label, confidence, probabilities, and summary features." />
            <NoteItem title="VLM tool" body="VlmInferenceAdapter.run(input) must return a structured diagnosis without leaking labels or raw file paths." />
            <NoteItem title="Reviewer" body="The reviewer keeps completed, needs_review, and rejected branches independent from model code." />
          </div>
        </section>
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
  return (
    <label className="uploadField">
      <div className="uploadIcon">{props.icon}</div>
      <div>
        <strong>{props.label}</strong>
        <span>{props.description}</span>
      </div>
      <div className="uploadButton">
        <UploadCloud size={16} />
        Select
      </div>
      <input
        accept={props.accept}
        className="fileInput"
        type="file"
        onChange={(event) => props.onChange(event.currentTarget.files?.item(0) ?? null)}
      />
    </label>
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
        <strong>Request failed</strong>
        <span>Check backend status, CORS, or uploaded files.</span>
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
        <strong>Awaiting diagnosis</strong>
        <span>Run the inspection workflow to populate the final decision.</span>
      </div>
    );
  }

  return (
    <div className="verdictBox">
      <div className={`status ${props.result.status}`}>{statusLabel[props.result.status]}</div>
      <h4>{props.result.diagnosis ?? "Diagnosis held"}</h4>
      <dl>
        <dt>Diagnosis ID</dt><dd>{props.result.diagnosis_id}</dd>
        <dt>Risk</dt><dd>{props.result.risk_level ?? "n/a"}</dd>
        <dt>Confidence</dt><dd>{props.result.confidence === null ? "n/a" : `${Math.round(props.result.confidence * 100)}%`}</dd>
        <dt>Route</dt><dd>{routeLabel[props.result.route]}</dd>
      </dl>
      <div className="reasonBox">
        <strong>Reason</strong>
        <p>{props.result.reason}</p>
      </div>
      <div className="reasonBox">
        <strong>Recommended action</strong>
        <p>{props.result.recommended_action ?? "Manual review required."}</p>
      </div>
    </div>
  );
}

function HistoryVerdict(props: { readonly item: DiagnosisListItem }) {
  return (
    <div className="verdictBox">
      <div className={`status ${props.item.status}`}>{statusLabel[props.item.status]}</div>
      <h4>{props.item.diagnosis ?? "Diagnosis held"}</h4>
      <dl>
        <dt>Diagnosis ID</dt><dd>{props.item.diagnosis_id}</dd>
        <dt>Risk</dt><dd>{props.item.risk_level ?? "n/a"}</dd>
        <dt>Confidence</dt><dd>{props.item.confidence === null ? "n/a" : `${Math.round(props.item.confidence * 100)}%`}</dd>
        <dt>Route</dt><dd>{routeLabel[props.item.route]}</dd>
      </dl>
      <div className="reasonBox">
        <strong>Reason</strong>
        <p>{props.item.reason}</p>
      </div>
    </div>
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
        <strong>No diagnosis history</strong>
        <span>Completed inspections will be listed here.</span>
      </div>
    );
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Diagnosis</th>
            <th>Status</th>
            <th>Route</th>
            <th>Risk</th>
            <th>Confidence</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {props.items.map((item) => (
            <tr className="clickableRow" key={item.diagnosis_id} onClick={() => props.onOpen(item)}>
              <td><strong>{item.diagnosis_id}</strong></td>
              <td><span className={`status ${item.status}`}>{statusLabel[item.status]}</span></td>
              <td>{routeLabel[item.route]}</td>
              <td>{item.risk_level ?? "n/a"}</td>
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
        <strong>Queue clear</strong>
        <span>No rejected or review-needed diagnosis is waiting.</span>
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
        <strong>No trace yet</strong>
        <span>Router, tool, reviewer, and report events will appear here.</span>
      </div>
    );
  }

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Kind</th>
            <th>Name</th>
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

function ModelStatusPanel(props: { readonly modelStatus: ModelRuntimeStatus | null }) {
  if (props.modelStatus === null) {
    return (
      <div className="emptyState compact">
        <Cpu size={24} />
        <strong>Runtime status pending</strong>
        <span>Backend model runtime information has not loaded yet.</span>
      </div>
    );
  }

  return (
    <div className="runtimeGrid">
      <RuntimeItem label="Agent mode" value={props.modelStatus.agent_mode} />
      <RuntimeItem label="Agents SDK" value={props.modelStatus.agents_sdk_installed ? "installed" : "not installed"} />
      <RuntimeItem label="Time-series model" value={`${props.modelStatus.time_series_model}@${props.modelStatus.time_series_version}`} />
      <RuntimeItem label="VLM model" value={`${props.modelStatus.vlm_model}@${props.modelStatus.vlm_version}`} />
    </div>
  );
}

function RuntimeItem(props: { readonly label: string; readonly value: string }) {
  return (
    <article className="runtimeItem">
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </article>
  );
}

function NoteItem(props: { readonly title: string; readonly body: string }) {
  return (
    <article className="noteItem">
      <p className="eyebrow">{props.title}</p>
      <span>{props.body}</span>
    </article>
  );
}

async function loadDashboard(): Promise<{
  backendStatus: BackendStatus;
  history: readonly DiagnosisListItem[];
  modelStatus: ModelRuntimeStatus | null;
  reviewQueue: readonly DiagnosisListItem[];
}> {
  const [health, history, reviewQueue, modelStatus] = await Promise.all([
    fetchHealth(),
    fetchDiagnosisHistory(),
    fetchReviewQueue(),
    fetchModelStatus(),
  ]);
  return {
    backendStatus: health.status === "ok" ? "online" : "offline",
    history: history.items,
    modelStatus,
    reviewQueue: reviewQueue.items,
  };
}

function backendStatusLabel(status: BackendStatus): string {
  if (status === "online") {
    return "online";
  }
  if (status === "offline") {
    return "offline";
  }
  return "checking";
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

function formatConfidence(confidence: number | null): string {
  return confidence === null ? "n/a" : `${Math.round(confidence * 100)}%`;
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
