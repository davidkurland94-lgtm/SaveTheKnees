import { useState, useRef, useCallback, useEffect } from "react";
import dicomParser from "dicom-parser";
import { api, type Study, type Prediction, type Label, type Report } from "./api";

// ─── DICOM parsing ────────────────────────────────────────────────────────────

type ParsedScan = {
  id: string;
  label: string;
  sequence: string;
  slice: string;
  imageData: ImageData;
  rows: number;
  cols: number;
  patientName: string;
  studyDate: string;
  patientAge: string;
  patientSex: string;
  accessionNumber: string;
  seriesDescription: string;
};

function formatStudyDate(raw: string): string {
  if (raw.length === 8) return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
  return raw || "Unknown";
}

async function parseDicomFile(file: File): Promise<ParsedScan | null> {
  try {
    const buffer = await file.arrayBuffer();
    const byteArray = new Uint8Array(buffer);
    const dataset = dicomParser.parseDicom(byteArray);

    const rows = dataset.uint16("x00280010") ?? 512;
    const cols = dataset.uint16("x00280011") ?? 512;
    const bitsAllocated = dataset.uint16("x00280100") ?? 16;
    const pixelRepresentation = dataset.uint16("x00280103") ?? 0;
    const wcStr = dataset.string("x00281050");
    const wwStr = dataset.string("x00281051");
    const windowCenter = wcStr ? parseFloat(wcStr.split("\\")[0]) : undefined;
    const windowWidth = wwStr ? parseFloat(wwStr.split("\\")[0]) : undefined;

    const pixelDataElement = dataset.elements["x7fe00010"];
    if (!pixelDataElement) return null;

    const pixelDataBuffer = buffer.slice(
      pixelDataElement.dataOffset,
      pixelDataElement.dataOffset + pixelDataElement.length
    );

    let pixelValues: ArrayLike<number>;
    let minVal = Infinity;
    let maxVal = -Infinity;

    if (bitsAllocated === 16) {
      const arr = pixelRepresentation === 1 ? new Int16Array(pixelDataBuffer) : new Uint16Array(pixelDataBuffer);
      pixelValues = arr;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
      }
    } else {
      pixelValues = new Uint8Array(pixelDataBuffer);
      minVal = 0; maxVal = 255;
    }

    const wc = windowCenter ?? (minVal + maxVal) / 2;
    const ww = windowWidth ?? (maxVal - minVal || 1);
    const lo = wc - ww / 2;

    const imageData = new ImageData(cols, rows);
    for (let i = 0; i < rows * cols; i++) {
      const v = pixelValues[i] ?? 0;
      const norm = Math.max(0, Math.min(255, Math.round(((v - lo) / ww) * 255)));
      const idx = i * 4;
      imageData.data[idx] = norm;
      imageData.data[idx + 1] = norm;
      imageData.data[idx + 2] = norm;
      imageData.data[idx + 3] = 255;
    }

    return {
      id: file.name,
      label: dataset.string("x0008103e") ?? file.name.replace(/\.dcm$/i, ""),
      sequence: dataset.string("x00080060") ?? "",
      slice: dataset.string("x00200013") ? `Slice ${dataset.string("x00200013")}` : file.name,
      imageData,
      rows,
      cols,
      patientName: (dataset.string("x00100010") ?? "").replace(/\^/g, " ").trim() || "Unknown Patient",
      studyDate: formatStudyDate(dataset.string("x00080020") ?? ""),
      patientAge: dataset.string("x00101010") ?? "",
      patientSex: dataset.string("x00100040") ?? "",
      accessionNumber: dataset.string("x00080050") ?? "",
      seriesDescription: dataset.string("x0008103e") ?? "",
    };
  } catch {
    return null;
  }
}

// ─── DicomCanvas ──────────────────────────────────────────────────────────────

function DicomCanvas({ imageData, brightness = 100, contrast = 100, className }: {
  imageData: ImageData;
  brightness?: number;
  contrast?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);
  }, [imageData]);
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
    />
  );
}

// ─── Shared UI atoms ──────────────────────────────────────────────────────────

function NavBar({ children }: { children?: React.ReactNode }) {
  return (
    <header className="border-b border-[#e9e4f8] px-6 py-4 flex items-center justify-between bg-white sticky top-0 z-10">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-[#7c3aed] flex items-center justify-center">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-[#1a1523]">KneeVision AI</div>
          <div className="text-xs text-[#6d5da8]">Musculoskeletal Imaging Platform</div>
        </div>
      </div>
      <div className="flex items-center gap-3">{children}</div>
    </header>
  );
}

function StatusBadge({ status }: { status: Study["status"] }) {
  const map = {
    pending: "bg-[#ede9fe] text-[#4c1d95] border border-[#ddd6fe]",
    urgent: "bg-red-50 text-red-700 border border-red-100",
    reviewed: "bg-emerald-50 text-emerald-700 border border-emerald-100",
  };
  const labels = { pending: "Pending Review", urgent: "Urgent", reviewed: "Reviewed" };
  return (
    <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${map[status]}`}>
      {labels[status]}
    </span>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, string> = {
    High: "bg-red-50 text-red-600 border border-red-100",
    Moderate: "bg-amber-50 text-amber-600 border border-amber-100",
    Low: "bg-[#f5f3ff] text-[#6d5da8] border border-[#e9e4f8]",
  };
  return <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${styles[severity]}`}>{severity}</span>;
}

function ProbabilityBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value > 0.7 ? "#ef4444" : value > 0.5 ? "#f59e0b" : "#a78bfa";
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-[#f5f3ff] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-semibold tabular-nums w-8 text-right" style={{ color }}>{pct}%</span>
    </div>
  );
}

// ─── Home Page ────────────────────────────────────────────────────────────────

type UploadMode = "single" | "sequence" | "folder";

function HomePage({
  onOpenStudy,
  onUpload,
}: {
  onOpenStudy: (study: Study) => void;
  onUpload: (files: File[]) => void;
}) {
  const [studies, setStudies] = useState<Study[]>([]);
  const [loadingStudies, setLoadingStudies] = useState(true);
  const [uploadMode, setUploadMode] = useState<UploadMode>("sequence");
  const [isDragging, setIsDragging] = useState(false);
  const [search, setSearch] = useState("");

  const singleRef = useRef<HTMLInputElement>(null);
  const sequenceRef = useRef<HTMLInputElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);

  // Attach webkitdirectory to folder input (not a standard React prop)
  useEffect(() => {
    folderRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  // Load studies from API on mount
  useEffect(() => {
    api.getStudies()
      .then(setStudies)
      .finally(() => setLoadingStudies(false));
  }, []);

  const handleFiles = useCallback(
    (files: File[]) => {
      const dcm = files.filter((f) => f.name.toLowerCase().endsWith(".dcm"));
      if (dcm.length) onUpload(dcm);
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(Array.from(e.dataTransfer.files));
    },
    [handleFiles]
  );

  const activeInputRef = uploadMode === "single" ? singleRef : uploadMode === "sequence" ? sequenceRef : folderRef;

  const filtered = studies.filter((s) =>
    s.patientName.toLowerCase().includes(search.toLowerCase()) ||
    s.accessionNumber.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="min-h-full bg-white flex flex-col">
      <NavBar>
        <div className="flex items-center gap-2 bg-[#faf9ff] border border-[#e9e4f8] rounded-full px-3 py-1.5">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
          </svg>
          <span className="text-xs font-semibold text-[#1a1523]">Dr. James Okafor</span>
        </div>
      </NavBar>

      <div className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 flex flex-col gap-8">

        {/* Upload zone */}
        <section>
          <h2 className="text-xl text-[#1a1523] mb-4" style={{ fontFamily: "'DM Serif Display', serif" }}>
            New Study
          </h2>

          {/* Mode tabs */}
          <div className="flex gap-1 bg-[#f5f3ff] rounded-xl p-1 w-fit mb-4">
            {(["single", "sequence", "folder"] as UploadMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setUploadMode(mode)}
                className={`px-4 py-1.5 rounded-lg text-xs font-semibold capitalize transition-all ${
                  uploadMode === mode
                    ? "bg-white text-[#7c3aed] shadow-sm"
                    : "text-[#6d5da8] hover:text-[#7c3aed]"
                }`}
              >
                {mode === "single" ? "Single image" : mode === "sequence" ? "Image sequence" : "Folder"}
              </button>
            ))}
          </div>

          {/* Hidden inputs */}
          <input ref={singleRef} type="file" accept=".dcm" className="hidden"
            onChange={(e) => handleFiles(Array.from(e.target.files ?? []))} />
          <input ref={sequenceRef} type="file" accept=".dcm" multiple className="hidden"
            onChange={(e) => handleFiles(Array.from(e.target.files ?? []))} />
          <input ref={folderRef} type="file" accept=".dcm" multiple className="hidden"
            onChange={(e) => handleFiles(Array.from(e.target.files ?? []))} />

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => activeInputRef.current?.click()}
            className={`rounded-2xl border-2 border-dashed cursor-pointer flex items-center justify-center gap-6 px-10 py-10 transition-all duration-200 select-none ${
              isDragging
                ? "border-[#7c3aed] bg-[#f5f3ff] shadow-lg shadow-purple-100 scale-[1.005]"
                : "border-[#c4b5fd] bg-[#faf9ff] hover:border-[#a78bfa] hover:bg-[#f5f3ff]"
            }`}
          >
            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 transition-colors ${
              isDragging ? "bg-[#ede9fe]" : "bg-white border border-[#e9e4f8]"
            }`}>
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke={isDragging ? "#7c3aed" : "#a78bfa"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                {uploadMode === "folder"
                  ? <><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></>
                  : uploadMode === "sequence"
                  ? <><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></>
                  : <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></>
                }
              </svg>
            </div>
            <div>
              <p className={`text-base font-semibold transition-colors ${isDragging ? "text-[#7c3aed]" : "text-[#1a1523]"}`}>
                {isDragging
                  ? "Release to load"
                  : uploadMode === "single" ? "Drop a single .dcm file"
                  : uploadMode === "sequence" ? "Drop multiple .dcm files"
                  : "Drop a folder of .dcm files"}
              </p>
              <p className="text-sm text-[#6d5da8] mt-0.5">
                or <span className="text-[#7c3aed] underline underline-offset-2">browse your computer</span>
                {" · "}
                <span className="font-mono text-xs">DICOM (.dcm) only</span>
              </p>
            </div>

            <div className="ml-auto flex gap-2 flex-wrap">
              {["Uncompressed", "16-bit", "Multi-frame"].map((t) => (
                <span key={t} className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-white border border-[#e9e4f8] text-[#6d5da8]">{t}</span>
              ))}
            </div>
          </div>
        </section>

        {/* Studies table */}
        <section className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl text-[#1a1523]" style={{ fontFamily: "'DM Serif Display', serif" }}>
              Studies
            </h2>
            {/* Search */}
            <div className="flex items-center gap-2 border border-[#e9e4f8] rounded-xl px-3 py-2 bg-[#faf9ff] w-60">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search patient or accession…"
                className="flex-1 text-xs bg-transparent text-[#1a1523] placeholder:text-[#b5a9d4] outline-none"
              />
            </div>
          </div>

          {loadingStudies ? (
            <div className="flex items-center gap-2 py-12 justify-center text-[#6d5da8] text-sm">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25"/>
                <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round"/>
              </svg>
              Loading studies from database…
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-12 text-center text-sm text-[#6d5da8]">No studies found.</div>
          ) : (
            <div className="rounded-2xl border border-[#e9e4f8] overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-[#faf9ff] border-b border-[#e9e4f8]">
                    {["Patient", "Date", "Accession", "Body Part", "Series", "Status", "Primary Finding", ""].map((h) => (
                      <th key={h} className="text-left text-xs font-semibold text-[#6d5da8] uppercase tracking-wide px-4 py-3">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((study, i) => (
                    <tr
                      key={study.id}
                      className={`border-b border-[#f5f3ff] hover:bg-[#faf9ff] transition-colors ${i === filtered.length - 1 ? "border-b-0" : ""}`}
                    >
                      <td className="px-4 py-3">
                        <div className="font-medium text-[#1a1523]">{study.patientName}</div>
                        <div className="text-xs text-[#6d5da8]">{[study.patientAge, study.patientSex].filter(Boolean).join(", ")}</div>
                      </td>
                      <td className="px-4 py-3 text-[#1a1523] tabular-nums">{study.studyDate}</td>
                      <td className="px-4 py-3 font-mono text-xs text-[#6d5da8]">{study.accessionNumber}</td>
                      <td className="px-4 py-3 text-[#6d5da8]">{study.bodyPart}</td>
                      <td className="px-4 py-3 text-[#6d5da8] tabular-nums">{study.seriesCount}</td>
                      <td className="px-4 py-3"><StatusBadge status={study.status} /></td>
                      <td className="px-4 py-3 text-xs text-[#6d5da8] max-w-[180px] truncate">
                        {study.primaryFinding ?? "—"}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => onOpenStudy(study)}
                          className="text-xs font-semibold text-[#7c3aed] hover:text-[#6d28d9] hover:underline"
                        >
                          Open →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ─── Processing screen ─────────────────────────────────────────────────────────

function ProcessingScreen({ fileCount, patientName }: { fileCount: number; patientName: string }) {
  const steps = [
    "Parsing DICOM metadata",
    "Reconstructing image series",
    "Running injury prediction model",   // ← calls api.predict()
    "Generating findings report",
  ];
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setCurrentStep((s) => Math.min(s + 1, steps.length - 1)), 700);
    return () => clearInterval(id);
  }, [steps.length]);

  return (
    <div className="min-h-full bg-white flex flex-col">
      <NavBar />
      <div className="flex-1 flex items-center justify-center px-6">
        <div className="w-full max-w-md flex flex-col items-center gap-8 text-center">
          <div className="relative w-20 h-20">
            <svg className="w-full h-full animate-spin" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r="34" fill="none" stroke="#ede9fe" strokeWidth="6" />
              <circle cx="40" cy="40" r="34" fill="none" stroke="#7c3aed" strokeWidth="6" strokeLinecap="round" strokeDasharray="50 164" />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5" strokeLinecap="round">
                <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" />
              </svg>
            </div>
          </div>
          <div>
            <h2 className="text-2xl text-[#1a1523] mb-2" style={{ fontFamily: "'DM Serif Display', serif" }}>
              Analysing Study
            </h2>
            <p className="text-sm text-[#6d5da8]">
              {fileCount} file{fileCount !== 1 ? "s" : ""}{patientName ? ` · ${patientName}` : ""}
            </p>
          </div>
          <div className="w-full flex flex-col gap-2">
            {steps.map((step, i) => {
              const done = i < currentStep;
              const active = i === currentStep;
              return (
                <div key={step} className={`flex items-center gap-3 rounded-xl px-4 py-3 border transition-all ${
                  active ? "bg-[#f5f3ff] border-[#c4b5fd]" : done ? "bg-white border-[#e9e4f8]" : "bg-white border-[#f0edfb] opacity-40"
                }`}>
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${done ? "bg-[#7c3aed]" : active ? "bg-[#ede9fe]" : "bg-[#f0edfb]"}`}>
                    {done
                      ? <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><polyline points="2 6 5 9 10 3" /></svg>
                      : active
                      ? <div className="w-2 h-2 rounded-full bg-[#7c3aed] animate-pulse" />
                      : <div className="w-2 h-2 rounded-full bg-[#c4b5fd]" />
                    }
                  </div>
                  <span className={`text-sm ${active ? "font-semibold text-[#7c3aed]" : done ? "text-[#1a1523]" : "text-[#b5a9d4]"}`}>
                    {step}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Viewer Page ──────────────────────────────────────────────────────────────

function ViewerPage({
  scans,
  study,
  predictions,
  labels,
  onBack,
  onReportSaved,
}: {
  scans: ParsedScan[];
  study: Study;
  predictions: Prediction[];
  labels: Label[];
  onBack: () => void;
  onReportSaved: (report: Report) => void;
}) {
  const [selectedScanId, setSelectedScanId] = useState(scans[0]?.id ?? null);
  const [opinion, setOpinion] = useState(study.report?.clinicalImpression ?? "");
  const [recommendation, setRecommendation] = useState(study.report?.recommendation ?? "");
  const [urgency, setUrgency] = useState<Report["urgency"]>(study.report?.urgency ?? "Routine");
  const [submitted, setSubmitted] = useState(study.report !== null);
  const [saving, setSaving] = useState(false);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [activeTab, setActiveTab] = useState<"predictions" | "opinion">("predictions");

  const selectedScan = scans.find((s) => s.id === selectedScanId) ?? scans[0];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!opinion.trim()) return;
    setSaving(true);
    const report: Report = {
      clinicalImpression: opinion,
      recommendation,
      urgency,
      doctorName: "Dr. James Okafor",
      submittedAt: new Date().toISOString(),
    };
    await api.saveReport(study.id, report); // ← API call: POST /studies/:id/report
    setSaving(false);
    setSubmitted(true);
    onReportSaved(report);
  };

  const topPrediction = predictions[0] ?? null;

  // Map predictions to their label metadata for description display
  const enrichedPredictions = predictions.map((p) => ({
    ...p,
    description: labels.find((l) => l.id === p.labelId)?.description ?? "",
  }));

  return (
    <div className="min-h-full bg-white flex flex-col">
      <NavBar>
        {study.patientName && (
          <div className="text-right">
            <div className="text-xs text-[#6d5da8]">Patient</div>
            <div className="text-sm font-semibold text-[#1a1523]">
              {[study.patientName, study.patientAge, study.patientSex].filter(Boolean).join(", ")}
            </div>
          </div>
        )}
        {study.studyDate && (
          <div className="text-right">
            <div className="text-xs text-[#6d5da8]">Study Date</div>
            <div className="text-sm font-semibold text-[#1a1523]">{study.studyDate}</div>
          </div>
        )}
        {study.accessionNumber && (
          <div className="text-right">
            <div className="text-xs text-[#6d5da8]">Accession</div>
            <div className="text-sm font-mono text-[#1a1523]">{study.accessionNumber}</div>
          </div>
        )}
        <StatusBadge status={study.status} />
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs font-medium text-[#6d5da8] hover:text-[#7c3aed] border border-[#e9e4f8] hover:border-[#c4b5fd] rounded-full px-3 py-1.5 transition-colors"
        >
          ← Back to studies
        </button>
      </NavBar>

      <div className="flex flex-1 min-h-0">
        {/* Scan Sidebar */}
        <aside className="w-44 border-r border-[#e9e4f8] bg-[#faf9ff] flex flex-col shrink-0">
          <div className="px-4 pt-4 pb-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#6d5da8]">
              Series · {scans.length}
            </p>
          </div>
          <div className="flex flex-col gap-1 px-2 pb-4 overflow-y-auto">
            {scans.map((scan) => (
              <button
                key={scan.id}
                onClick={() => setSelectedScanId(scan.id)}
                className={`w-full text-left rounded-lg overflow-hidden border transition-all duration-150 ${
                  selectedScan?.id === scan.id
                    ? "border-[#a78bfa] shadow-md shadow-purple-100"
                    : "border-transparent hover:border-[#e9e4f8]"
                }`}
              >
                <div className="relative bg-black" style={{ aspectRatio: `${scan.cols}/${scan.rows}` }}>
                  <DicomCanvas imageData={scan.imageData} className="w-full h-full object-contain" />
                  {selectedScan?.id === scan.id && (
                    <div className="absolute inset-0 ring-2 ring-inset ring-[#a78bfa]" />
                  )}
                </div>
                <div className="px-2 py-1.5 bg-white">
                  <p className="text-xs font-semibold text-[#1a1523] truncate">{scan.label || scan.sequence}</p>
                  <p className="text-[10px] text-[#6d5da8] truncate">{scan.slice}</p>
                </div>
              </button>
            ))}
          </div>
        </aside>

        {/* Main viewer */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#080810]">
          <div className="flex-1 flex items-start justify-center p-6 min-h-0">
            {selectedScan && (
              <div className="relative max-w-2xl w-full" style={{ aspectRatio: `${selectedScan.cols}/${selectedScan.rows}` }}>
                <DicomCanvas
                  imageData={selectedScan.imageData}
                  brightness={brightness}
                  contrast={contrast}
                  className="w-full h-full rounded-sm"
                />
                <div className="absolute top-3 left-3 flex flex-col gap-1">
                  <span className="text-xs font-mono text-[#a78bfa] bg-black/60 px-2 py-0.5 rounded">{selectedScan.sequence || "DICOM"}</span>
                  <span className="text-xs font-mono text-white/60 bg-black/60 px-2 py-0.5 rounded">{selectedScan.cols} × {selectedScan.rows}</span>
                </div>
                <div className="absolute top-3 right-3">
                  <span className="text-xs font-mono text-white/40 bg-black/60 px-2 py-0.5 rounded">{selectedScan.slice}</span>
                </div>
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-20">
                  <div className="w-full h-px bg-[#a78bfa]" />
                </div>
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-20">
                  <div className="h-full w-px bg-[#a78bfa]" />
                </div>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="border-t border-white/5 bg-[#0d0d18] px-6 py-3 flex items-center gap-8">
            <div className="flex items-center gap-3 flex-1">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
              </svg>
              <span className="text-xs text-white/40 w-20">Brightness</span>
              <input type="range" min={50} max={200} value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} className="flex-1 accent-[#a78bfa] h-1" />
              <span className="text-xs font-mono text-white/30 w-8 text-right">{brightness}</span>
            </div>
            <div className="flex items-center gap-3 flex-1">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="10" /><path d="M12 2a10 10 0 0 1 0 20z" fill="#a78bfa" />
              </svg>
              <span className="text-xs text-white/40 w-20">Contrast</span>
              <input type="range" min={50} max={250} value={contrast} onChange={(e) => setContrast(Number(e.target.value))} className="flex-1 accent-[#a78bfa] h-1" />
              <span className="text-xs font-mono text-white/30 w-8 text-right">{contrast}</span>
            </div>
            <button onClick={() => { setBrightness(100); setContrast(100); }} className="text-xs text-[#6d5da8] hover:text-[#a78bfa] transition-colors">
              Reset
            </button>
          </div>
        </main>

        {/* Right panel */}
        <aside className="w-80 border-l border-[#e9e4f8] bg-white flex flex-col shrink-0">
          <div className="bg-[#faf9ff] border-b border-[#e9e4f8] px-5 py-4">
            <h2 className="text-base font-semibold text-[#1a1523]" style={{ fontFamily: "'DM Serif Display', serif" }}>
              {selectedScan?.seriesDescription || study.bodyPart || "MRI Study"}
            </h2>
            <p className="text-xs text-[#6d5da8] mt-0.5">
              {scans.length} series · {selectedScan?.cols} × {selectedScan?.rows}px
            </p>
            <div className="mt-3 flex gap-2 flex-wrap">
              {[selectedScan?.sequence, study.patientAge ? `Age ${study.patientAge}` : null, study.patientSex]
                .filter(Boolean)
                .map((tag) => (
                  <span key={tag} className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#ede9fe] text-[#4c1d95]">{tag}</span>
                ))}
            </div>
          </div>

          <div className="flex border-b border-[#e9e4f8]">
            {(["predictions", "opinion"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-3 text-xs font-semibold capitalize tracking-wide transition-colors ${
                  activeTab === tab ? "text-[#7c3aed] border-b-2 border-[#7c3aed]" : "text-[#6d5da8] hover:text-[#7c3aed]"
                }`}
              >
                {tab === "predictions" ? "AI Predictions" : "Doctor Opinion"}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto">
            {activeTab === "predictions" ? (
              <div className="px-5 py-5 flex flex-col gap-5">
                <div className="flex items-start gap-2">
                  <div className="w-1 min-h-8 rounded-full bg-gradient-to-b from-[#7c3aed] to-[#a78bfa] shrink-0" />
                  <p className="text-xs text-[#6d5da8] leading-relaxed">
                    Predictions are returned by the inference API and reflect the model's confidence across all loaded series. Advisory only.
                  </p>
                </div>

                {/* Labels loaded from API */}
                <div className="flex flex-col gap-4">
                  {enrichedPredictions.map((p) => (
                    <div key={p.labelId} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-[#1a1523]">{p.label}</span>
                        <SeverityBadge severity={p.severity} />
                      </div>
                      <ProbabilityBar value={p.probability} />
                      {p.description && (
                        <p className="text-[10px] text-[#6d5da8]">{p.description}</p>
                      )}
                    </div>
                  ))}
                </div>

                {topPrediction && topPrediction.probability > 0.5 && (
                  <div className="rounded-xl bg-red-50 border border-red-100 p-4">
                    <p className="text-xs font-semibold text-red-700 uppercase tracking-wide mb-1">Primary Finding</p>
                    <p className="text-sm font-medium text-red-800">
                      {topPrediction.label} — {Math.round(topPrediction.probability * 100)}%
                    </p>
                    <p className="text-xs text-red-600 mt-1 leading-relaxed">
                      {enrichedPredictions[0]?.description}
                    </p>
                  </div>
                )}

                <button
                  onClick={() => setActiveTab("opinion")}
                  className="w-full py-2.5 rounded-xl bg-[#7c3aed] text-white text-sm font-semibold hover:bg-[#6d28d9] transition-colors"
                >
                  {submitted ? "View Doctor Opinion →" : "Add Doctor Opinion →"}
                </button>
              </div>
            ) : submitted ? (
              <div className="px-5 py-8 flex flex-col items-center text-center gap-4">
                <div className="w-14 h-14 rounded-full bg-[#ede9fe] flex items-center justify-center">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-base font-semibold text-[#1a1523]" style={{ fontFamily: "'DM Serif Display', serif" }}>Opinion Submitted</h3>
                  <p className="text-xs text-[#6d5da8] mt-1 leading-relaxed">
                    Saved via <span className="font-mono">POST /studies/{study.id}/report</span>
                  </p>
                </div>
                <div className="w-full rounded-xl bg-[#faf9ff] border border-[#e9e4f8] p-4 text-left">
                  <p className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide mb-1">Clinical Impression</p>
                  <p className="text-sm text-[#1a1523] leading-relaxed">{opinion}</p>
                  {recommendation && (
                    <>
                      <p className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide mt-3 mb-1">Recommendation</p>
                      <p className="text-sm text-[#1a1523] leading-relaxed">{recommendation}</p>
                    </>
                  )}
                  <p className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide mt-3 mb-1">Urgency</p>
                  <p className="text-sm text-[#1a1523]">{urgency}</p>
                </div>
                <button onClick={() => setSubmitted(false)} className="text-xs text-[#7c3aed] hover:underline">
                  Revise opinion
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="px-5 py-5 flex flex-col gap-4">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-7 h-7 rounded-full bg-[#ede9fe] flex items-center justify-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-[#1a1523]">Dr. James Okafor</p>
                    <p className="text-[10px] text-[#6d5da8]">Orthopedic Surgeon</p>
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide">Clinical Impression</label>
                  <textarea
                    value={opinion}
                    onChange={(e) => setOpinion(e.target.value)}
                    placeholder="Describe your clinical findings based on the imaging…"
                    rows={5}
                    required
                    className="w-full rounded-xl border border-[#e9e4f8] bg-[#faf9ff] px-3.5 py-3 text-sm text-[#1a1523] placeholder:text-[#b5a9d4] focus:outline-none focus:ring-2 focus:ring-[#a78bfa] focus:border-transparent resize-none leading-relaxed transition-shadow"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide">Treatment Recommendation</label>
                  <textarea
                    value={recommendation}
                    onChange={(e) => setRecommendation(e.target.value)}
                    placeholder="e.g. Surgical intervention, physiotherapy, further imaging…"
                    rows={3}
                    className="w-full rounded-xl border border-[#e9e4f8] bg-[#faf9ff] px-3.5 py-3 text-sm text-[#1a1523] placeholder:text-[#b5a9d4] focus:outline-none focus:ring-2 focus:ring-[#a78bfa] focus:border-transparent resize-none leading-relaxed transition-shadow"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide">Urgency</label>
                  <div className="flex gap-2">
                    {(["Routine", "Urgent", "Emergency"] as Report["urgency"][]).map((u) => (
                      <button
                        key={u}
                        type="button"
                        onClick={() => setUrgency(u)}
                        className={`flex-1 text-center text-xs font-medium py-2 rounded-lg border transition-all ${
                          urgency === u
                            ? "border-[#7c3aed] bg-[#ede9fe] text-[#4c1d95]"
                            : "border-[#e9e4f8] text-[#6d5da8] hover:border-[#c4b5fd]"
                        }`}
                      >
                        {u}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={saving}
                  className="w-full py-3 rounded-xl bg-[#7c3aed] text-white text-sm font-semibold hover:bg-[#6d28d9] active:scale-95 transition-all mt-1 disabled:opacity-60"
                >
                  {saving ? "Saving…" : "Submit Opinion"}
                </button>
                <p className="text-[10px] text-center text-[#b5a9d4] leading-relaxed">
                  Submitted via <span className="font-mono">POST /studies/{study.id}/report</span>
                </p>
              </form>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

type AppPage = "home" | "processing" | "viewer";

export default function App() {
  const [page, setPage] = useState<AppPage>("home");
  const [scans, setScans] = useState<ParsedScan[]>([]);
  const [activeStudy, setActiveStudy] = useState<Study | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [labels, setLabels] = useState<Label[]>([]);
  const [processingFileCount, setProcessingFileCount] = useState(0);
  const [processingPatientName, setProcessingPatientName] = useState("");

  // Load labels from API on mount
  useEffect(() => {
    api.getLabels().then(setLabels); // ← GET /labels
  }, []);

  const handleUpload = async (files: File[]) => {
    setProcessingFileCount(files.length);
    setProcessingPatientName("");
    setPage("processing");

    // Parse DICOM files locally
    const results = await Promise.all(files.map(parseDicomFile));
    const parsed = results.filter((s): s is ParsedScan => s !== null);

    if (parsed.length === 0) {
      setPage("home");
      return;
    }

    setProcessingPatientName(parsed[0].patientName);

    // Call prediction API
    const preds = await api.predict(files); // ← POST /predict

    // Create study record
    const study = await api.createStudy(files, parsed[0].patientName, preds); // ← POST /studies

    setScans(parsed);
    setPredictions(preds);
    setActiveStudy(study);
    setPage("viewer");
  };

  const handleOpenStudy = (study: Study) => {
    // Existing studies from DB: predictions already embedded in study record.
    // DICOM files would be fetched from the server here:
    // TODO: const files = await api.getStudyFiles(study.id);  // GET /studies/:id/files
    // TODO: const parsed = await Promise.all(files.map(parseDicomFile));
    // For now, show the viewer with empty scans (no local files available).
    setScans([]);
    setPredictions(study.predictions ?? []);
    setActiveStudy(study);
    setPage("viewer");
  };

  const handleReportSaved = (report: Report) => {
    if (!activeStudy) return;
    setActiveStudy({ ...activeStudy, report, status: "reviewed" });
  };

  if (page === "processing") {
    return <ProcessingScreen fileCount={processingFileCount} patientName={processingPatientName} />;
  }

  if (page === "viewer" && activeStudy) {
    return (
      <ViewerPage
        scans={scans}
        study={activeStudy}
        predictions={predictions}
        labels={labels}
        onBack={() => setPage("home")}
        onReportSaved={handleReportSaved}
      />
    );
  }

  return <HomePage onOpenStudy={handleOpenStudy} onUpload={handleUpload} />;
}
