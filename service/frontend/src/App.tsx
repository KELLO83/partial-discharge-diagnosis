import { Activity, FileImage, FileText, PanelLeft, Send, UploadCloud, Zap } from "lucide-react";
import { useMemo, useState } from "react";

import { submitDiagnosis } from "./api";
import { fileDisplayText } from "./fileDisplay";
import { buildInputPresence } from "./formState";
import { DEFAULT_METADATA } from "./metadataDefaults";
import { selectInputRoute } from "./route";
import type { DiagnosisResponse, MetadataForm } from "./types";
import "./styles.css";

export function App() {
  const [image, setImage] = useState<File | null>(null);
  const [csv, setCsv] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<MetadataForm>(DEFAULT_METADATA);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [status, setStatus] = useState<"idle" | "submitting" | "failed">("idle");
  const route = useMemo(
    () => selectInputRoute(buildInputPresence({ hasImage: image !== null, hasTimeseries: csv !== null, metadata })),
    [image, csv, metadata],
  );

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setStatus("submitting");
    try {
      const response = await submitDiagnosis({ image, csv, metadata });
      setResult(response);
      setStatus("idle");
    } catch (error) {
      setStatus("failed");
      throw error;
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Zap aria-hidden="true" />
          <strong>PD Whisper</strong>
        </div>
        <nav className="nav">
          <div className="nav-item active">
            <Activity aria-hidden="true" />
            <span>Diagnosis</span>
          </div>
        </nav>
        <div className="profile">
          <div>PD</div>
          <span>Agent Workflow</span>
        </div>
      </aside>
      <section className="content">
        <div className="topbar">
          <PanelLeft aria-hidden="true" />
          <div>
            <p>Partial Discharge Diagnosis</p>
            <h1>PRPD 이미지와 시계열 CSV를 입력해 추론 워크플로우를 실행합니다.</h1>
          </div>
        </div>
        <div className="grid-shell">
          <form className="diagnosis-form" onSubmit={(event) => void handleSubmit(event)}>
            <div className="section-title">
              <span>Input</span>
              <div className="route-pill">
                <Activity aria-hidden="true" />
                <span>{route}</span>
              </div>
            </div>
            <div className="upload-row">
              <label className="upload-box">
                <div className="upload-icon">
                  <FileImage aria-hidden="true" />
                </div>
                <span>PRPD PNG</span>
                <input
                  accept="image/png"
                  className="file-input"
                  type="file"
                  onChange={(event) => setImage(event.currentTarget.files?.item(0) ?? null)}
                />
                <div className="file-control">
                  <span className="file-control-button">
                    <UploadCloud aria-hidden="true" />
                    이미지 선택
                  </span>
                  <small>{fileDisplayText(image, "PRPD PNG")}</small>
                </div>
              </label>
              <label className="upload-box">
                <div className="upload-icon">
                  <FileText aria-hidden="true" />
                </div>
                <span>시계열 CSV</span>
                <input
                  accept=".csv,text/csv"
                  className="file-input"
                  type="file"
                  onChange={(event) => setCsv(event.currentTarget.files?.item(0) ?? null)}
                />
                <div className="file-control">
                  <span className="file-control-button">
                    <UploadCloud aria-hidden="true" />
                    CSV 선택
                  </span>
                  <small>{fileDisplayText(csv, "시계열 CSV")}</small>
                </div>
              </label>
            </div>
            <div className="metadata-grid">
              <Field label="설비명" value={metadata.equipmentName} onChange={(value) => setMetadata({ ...metadata, equipmentName: value })} />
              <Field label="정격 전압" value={metadata.ratedVoltage} onChange={(value) => setMetadata({ ...metadata, ratedVoltage: value })} />
              <Field label="정격 전류" value={metadata.ratedCurrent} onChange={(value) => setMetadata({ ...metadata, ratedCurrent: value })} />
              <Field label="센서 타입" value={metadata.sensorType} onChange={(value) => setMetadata({ ...metadata, sensorType: value })} />
              <Field label="온도" value={metadata.temperature} onChange={(value) => setMetadata({ ...metadata, temperature: value })} />
              <Field label="습도" value={metadata.humidity} onChange={(value) => setMetadata({ ...metadata, humidity: value })} />
            </div>
            <div className="action-row">
              <button type="submit" disabled={status === "submitting"}>
                <Send aria-hidden="true" />
                {status === "submitting" ? "진단 중" : "진단 실행"}
              </button>
            </div>
          </form>
          <ResultPanel result={result} status={status} />
        </div>
      </section>
    </main>
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

function ResultPanel(props: { readonly result: DiagnosisResponse | null; readonly status: "idle" | "submitting" | "failed" }) {
  if (props.status === "failed") {
    return <aside className="result-panel error">요청 실패. FastAPI 서버 상태를 확인하세요.</aside>;
  }
  if (props.result === null) {
    return <aside className="result-panel empty">입력 조합에 따라 VLM-only, timeseries-only, hybrid workflow가 선택됩니다.</aside>;
  }
  return (
    <aside className="result-panel">
      <div>
        <span className="status">{props.result.status}</span>
        <strong>{props.result.diagnosis ?? "진단 보류"}</strong>
      </div>
      <p>{props.result.reason}</p>
      <dl>
        <dt>route</dt>
        <dd>{props.result.route}</dd>
        <dt>confidence</dt>
        <dd>{props.result.confidence ?? "-"}</dd>
        <dt>trace</dt>
        <dd>{props.result.trace_id}</dd>
      </dl>
    </aside>
  );
}
