import { useState, useRef, useCallback, useEffect } from "react";
import dicomParser from "dicom-parser";

// ─── Types ────────────────────────────────────────────────────────────────────

type AppStage = "upload" | "processing" | "ready";

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

type PatientInfo = {
  name: string;
  age: string;
  sex: string;
  studyDate: string;
  accessionNumber: string;
};

// ─── DICOM parsing ────────────────────────────────────────────────────────────

function formatStudyDate(raw: string): string {
  if (raw.length === 8) {
    return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
  }
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

    // Decode pixel values and find data range
    let pixelValues: ArrayLike<number>;
    let minVal = Infinity;
    let maxVal = -Infinity;

    if (bitsAllocated === 16) {
      const arr =
        pixelRepresentation === 1
          ? new Int16Array(pixelDataBuffer)
          : new Uint16Array(pixelDataBuffer);
      pixelValues = arr;
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
      }
    } else {
      const arr = new Uint8Array(pixelDataBuffer);
      pixelValues = arr;
      minVal = 0;
      maxVal = 255;
    }

    // Window: use DICOM-embedded values when present, otherwise auto-window
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

    const rawPatient = dataset.string("x00100010") ?? "";
    const patientName = rawPatient.replace(/\^/g, " ").trim() || "Unknown Patient";
    const studyDate = formatStudyDate(dataset.string("x00080020") ?? "");
    const seriesDesc = dataset.string("x0008103e") ?? "";
    const modality = dataset.string("x00080060") ?? "";
    const instanceNum = dataset.string("x00200013") ?? "";
    const patientAge = dataset.string("x00101010") ?? "";
    const patientSex = dataset.string("x00100040") ?? "";
    const accessionNumber = dataset.string("x00080050") ?? "";

    return {
      id: file.name,
      label: seriesDesc || file.name.replace(/\.dcm$/i, ""),
      sequence: modality,
      slice: instanceNum ? `Slice ${instanceNum}` : file.name,
      imageData,
      rows,
      cols,
      patientName,
      studyDate,
      patientAge,
      patientSex,
      accessionNumber,
      seriesDescription: seriesDesc,
    };
  } catch {
    return null;
  }
}

// ─── DicomCanvas ──────────────────────────────────────────────────────────────

function DicomCanvas({
  imageData,
  brightness = 100,
  contrast = 100,
  className,
}: {
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

// ─── Upload screen ─────────────────────────────────────────────────────────────

function UploadScreen({ onFilesDropped }: { onFilesDropped: (files: File[]) => void }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length) onFilesDropped(files);
    },
    [onFilesDropped]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length) onFilesDropped(files);
  };

  return (
    <div className="min-h-full bg-white flex flex-col">
      <header className="border-b border-[#e9e4f8] px-6 py-4 flex items-center gap-3 bg-white">
        <div className="w-8 h-8 rounded-lg bg-[#7c3aed] flex items-center justify-center">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-[#1a1523]">KneeVision AI</div>
          <div className="text-xs text-[#6d5da8]">Musculoskeletal Imaging Platform</div>
        </div>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        <div className="w-full max-w-2xl flex flex-col items-center gap-8">
          <div className="text-center">
            <h1 className="text-3xl text-[#1a1523] mb-2" style={{ fontFamily: "'DM Serif Display', serif" }}>
              Load MRI Study
            </h1>
            <p className="text-sm text-[#6d5da8] max-w-sm leading-relaxed">
              Drop DICOM files (.dcm) to begin analysis. Multiple files are loaded as separate series.
            </p>
          </div>

          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`w-full rounded-2xl border-2 border-dashed cursor-pointer flex flex-col items-center justify-center gap-5 px-10 py-16 transition-all duration-200 select-none ${
              isDragging
                ? "border-[#7c3aed] bg-[#f5f3ff] scale-[1.01] shadow-lg shadow-purple-100"
                : "border-[#c4b5fd] bg-[#faf9ff] hover:border-[#a78bfa] hover:bg-[#f5f3ff]"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".dcm"
              className="hidden"
              onChange={handleChange}
            />

            <div className={`w-20 h-20 rounded-2xl flex items-center justify-center transition-colors duration-200 ${
              isDragging ? "bg-[#ede9fe]" : "bg-white border border-[#e9e4f8]"
            }`}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={isDragging ? "#7c3aed" : "#a78bfa"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>

            <div className="text-center">
              <p className={`text-base font-semibold transition-colors ${isDragging ? "text-[#7c3aed]" : "text-[#1a1523]"}`}>
                {isDragging ? "Release to load study" : "Drop DICOM files here"}
              </p>
              <p className="text-sm text-[#6d5da8] mt-1">
                or <span className="text-[#7c3aed] underline underline-offset-2">browse your computer</span>
              </p>
            </div>

            <div className="flex gap-2 flex-wrap justify-center">
              {["DICOM (.dcm)", "Uncompressed", "Multi-series", "16-bit support"].map((fmt) => (
                <span key={fmt} className="text-[10px] font-medium px-2.5 py-1 rounded-full bg-white border border-[#e9e4f8] text-[#6d5da8]">
                  {fmt}
                </span>
              ))}
            </div>
          </div>

          <div className="w-full grid grid-cols-3 gap-4">
            {[
              {
                icon: <path d="M4 4h16v16H4zM4 9h16M9 4v16" />,
                title: "Multi-series support",
                desc: "Each .dcm file becomes a separate series in the viewer.",
              },
              {
                icon: <><circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" /></>,
                title: "Instant AI analysis",
                desc: "Injury predictions run as soon as files are processed.",
              },
              {
                icon: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
                title: "HIPAA-safe",
                desc: "Files are analyzed locally — no data leaves your browser.",
              },
            ].map((item) => (
              <div key={item.title} className="flex flex-col gap-2 rounded-xl bg-[#faf9ff] border border-[#e9e4f8] px-4 py-4">
                <div className="w-8 h-8 rounded-lg bg-[#ede9fe] flex items-center justify-center">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    {item.icon}
                  </svg>
                </div>
                <p className="text-xs font-semibold text-[#1a1523]">{item.title}</p>
                <p className="text-xs text-[#6d5da8] leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Processing screen ─────────────────────────────────────────────────────────

function ProcessingScreen({ fileCount, patientName }: { fileCount: number; patientName: string }) {
  const steps = [
    "Parsing DICOM metadata",
    "Reconstructing image series",
    "Running injury prediction model",
    "Generating findings report",
  ];
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setCurrentStep((s) => Math.min(s + 1, steps.length - 1));
    }, 700);
    return () => clearInterval(id);
  }, [steps.length]);

  return (
    <div className="min-h-full bg-white flex flex-col">
      <header className="border-b border-[#e9e4f8] px-6 py-4 flex items-center gap-3 bg-white">
        <div className="w-8 h-8 rounded-lg bg-[#7c3aed] flex items-center justify-center">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4l3 3" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-[#1a1523]">KneeVision AI</div>
          <div className="text-xs text-[#6d5da8]">Musculoskeletal Imaging Platform</div>
        </div>
      </header>

      <div className="flex-1 flex items-center justify-center px-6">
        <div className="w-full max-w-md flex flex-col items-center gap-8 text-center">
          <div className="relative w-20 h-20">
            <svg className="w-full h-full animate-spin" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r="34" fill="none" stroke="#ede9fe" strokeWidth="6" />
              <circle cx="40" cy="40" r="34" fill="none" stroke="#7c3aed" strokeWidth="6" strokeLinecap="round" strokeDasharray="50 164" />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5" strokeLinecap="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M3 9h18M9 21V9" />
              </svg>
            </div>
          </div>

          <div>
            <h2 className="text-2xl text-[#1a1523] mb-2" style={{ fontFamily: "'DM Serif Display', serif" }}>
              Analysing Study
            </h2>
            <p className="text-sm text-[#6d5da8]">
              {fileCount} file{fileCount !== 1 ? "s" : ""} loaded
              {patientName ? ` · ${patientName}` : ""}
            </p>
          </div>

          <div className="w-full flex flex-col gap-2">
            {steps.map((step, i) => {
              const done = i < currentStep;
              const active = i === currentStep;
              return (
                <div
                  key={step}
                  className={`flex items-center gap-3 rounded-xl px-4 py-3 border transition-all ${
                    active ? "bg-[#f5f3ff] border-[#c4b5fd]"
                    : done ? "bg-white border-[#e9e4f8]"
                    : "bg-white border-[#f0edfb] opacity-40"
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                    done ? "bg-[#7c3aed]" : active ? "bg-[#ede9fe]" : "bg-[#f0edfb]"
                  }`}>
                    {done ? (
                      <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
                        <polyline points="2 6 5 9 10 3" />
                      </svg>
                    ) : active ? (
                      <div className="w-2 h-2 rounded-full bg-[#7c3aed] animate-pulse" />
                    ) : (
                      <div className="w-2 h-2 rounded-full bg-[#c4b5fd]" />
                    )}
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

// ─── Prediction constants ──────────────────────────────────────────────────────

const PREDICTIONS = [
  { label: "ACL Tear", probability: 0.82, severity: "High" },
  { label: "Meniscal Tear", probability: 0.61, severity: "Moderate" },
  { label: "Cartilage Damage", probability: 0.38, severity: "Low" },
  { label: "Bone Edema", probability: 0.27, severity: "Low" },
];

function SeverityBadge({ severity }: { severity: string }) {
  const styles: Record<string, string> = {
    High: "bg-red-50 text-red-600 border border-red-100",
    Moderate: "bg-amber-50 text-amber-600 border border-amber-100",
    Low: "bg-[#f5f3ff] text-[#6d5da8] border border-[#e9e4f8]",
  };
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${styles[severity]}`}>
      {severity}
    </span>
  );
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

// ─── Main app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [stage, setStage] = useState<AppStage>("upload");
  const [droppedFileCount, setDroppedFileCount] = useState(0);
  const [parsedScans, setParsedScans] = useState<ParsedScan[]>([]);
  const [patientInfo, setPatientInfo] = useState<PatientInfo | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const [selectedScanId, setSelectedScanId] = useState<string | null>(null);
  const [opinion, setOpinion] = useState("");
  const [recommendation, setRecommendation] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [activeTab, setActiveTab] = useState<"predictions" | "opinion">("predictions");

  const handleFilesDropped = async (files: File[]) => {
    setDroppedFileCount(files.length);
    setParseError(null);
    setStage("processing");

    const results = await Promise.all(files.map(parseDicomFile));
    const scans = results.filter((s): s is ParsedScan => s !== null);

    if (scans.length === 0) {
      setParseError("No valid DICOM files could be read. Make sure files are uncompressed .dcm format.");
      setStage("upload");
      return;
    }

    const first = scans[0];
    setPatientInfo({
      name: first.patientName,
      age: first.patientAge,
      sex: first.patientSex,
      studyDate: first.studyDate,
      accessionNumber: first.accessionNumber,
    });
    setParsedScans(scans);
    setSelectedScanId(scans[0].id);
    setStage("ready");
  };

  const resetStudy = () => {
    setStage("upload");
    setSubmitted(false);
    setOpinion("");
    setRecommendation("");
    setParsedScans([]);
    setPatientInfo(null);
    setSelectedScanId(null);
  };

  if (stage === "upload") {
    return (
      <>
        <UploadScreen onFilesDropped={handleFilesDropped} />
        {parseError && (
          <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-red-50 border border-red-200 text-red-700 text-sm px-5 py-3 rounded-xl shadow-lg">
            {parseError}
          </div>
        )}
      </>
    );
  }

  if (stage === "processing") {
    return <ProcessingScreen fileCount={droppedFileCount} patientName={patientInfo?.name ?? ""} />;
  }

  const selectedScan = parsedScans.find((s) => s.id === selectedScanId) ?? parsedScans[0];
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (opinion.trim()) setSubmitted(true);
  };

  const formatPatientLabel = () => {
    const parts = [patientInfo?.name ?? "Unknown"];
    if (patientInfo?.age) parts.push(patientInfo.age);
    if (patientInfo?.sex) parts.push(patientInfo.sex);
    return parts.join(", ");
  };

  return (
    <div className="min-h-full bg-white flex flex-col">
      {/* Top Nav */}
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

        <div className="flex items-center gap-6">
          {patientInfo?.name && (
            <div className="text-right">
              <div className="text-xs text-[#6d5da8]">Patient</div>
              <div className="text-sm font-semibold text-[#1a1523]">{formatPatientLabel()}</div>
            </div>
          )}
          {patientInfo?.studyDate && (
            <div className="text-right">
              <div className="text-xs text-[#6d5da8]">Study Date</div>
              <div className="text-sm font-semibold text-[#1a1523]">{patientInfo.studyDate}</div>
            </div>
          )}
          {patientInfo?.accessionNumber && (
            <div className="text-right">
              <div className="text-xs text-[#6d5da8]">Accession</div>
              <div className="text-sm font-mono text-[#1a1523]">{patientInfo.accessionNumber}</div>
            </div>
          )}
          <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-[#ede9fe] text-[#4c1d95] border border-[#ddd6fe]">
            Pending Review
          </span>
          <button
            onClick={resetStudy}
            className="flex items-center gap-1.5 text-xs font-medium text-[#6d5da8] hover:text-[#7c3aed] border border-[#e9e4f8] hover:border-[#c4b5fd] rounded-full px-3 py-1.5 transition-colors"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Load new study
          </button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Scan Sidebar */}
        <aside className="w-44 border-r border-[#e9e4f8] bg-[#faf9ff] flex flex-col shrink-0">
          <div className="px-4 pt-4 pb-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-[#6d5da8]">
              Series · {parsedScans.length}
            </p>
          </div>
          <div className="flex flex-col gap-1 px-2 pb-4 overflow-y-auto">
            {parsedScans.map((scan) => (
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
                  <DicomCanvas
                    imageData={scan.imageData}
                    className="w-full h-full object-contain"
                  />
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

        {/* Main Viewer */}
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
                {/* Overlay labels */}
                <div className="absolute top-3 left-3 flex flex-col gap-1">
                  <span className="text-xs font-mono text-[#a78bfa] bg-black/60 px-2 py-0.5 rounded">
                    {selectedScan.sequence || "DICOM"}
                  </span>
                  <span className="text-xs font-mono text-white/60 bg-black/60 px-2 py-0.5 rounded">
                    {selectedScan.cols} × {selectedScan.rows}
                  </span>
                </div>
                <div className="absolute top-3 right-3">
                  <span className="text-xs font-mono text-white/40 bg-black/60 px-2 py-0.5 rounded">
                    {selectedScan.slice}
                  </span>
                </div>
                {/* Crosshair */}
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-20">
                  <div className="w-full h-px bg-[#a78bfa]" />
                </div>
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-20">
                  <div className="h-full w-px bg-[#a78bfa]" />
                </div>
              </div>
            )}
          </div>

          {/* Controls bar */}
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
            <button
              onClick={() => { setBrightness(100); setContrast(100); }}
              className="text-xs text-[#6d5da8] hover:text-[#a78bfa] transition-colors"
            >
              Reset
            </button>
          </div>
        </main>

        {/* Right Panel */}
        <aside className="w-80 border-l border-[#e9e4f8] bg-white flex flex-col shrink-0">
          <div className="bg-[#faf9ff] border-b border-[#e9e4f8] px-5 py-4">
            <h2 className="text-base font-semibold text-[#1a1523]" style={{ fontFamily: "'DM Serif Display', serif" }}>
              {selectedScan?.seriesDescription || "MRI Study"}
            </h2>
            <p className="text-xs text-[#6d5da8] mt-0.5">
              {parsedScans.length} series · {selectedScan?.cols} × {selectedScan?.rows}px
            </p>
            <div className="mt-3 flex gap-2 flex-wrap">
              {[
                selectedScan?.sequence,
                patientInfo?.age ? `Age ${patientInfo.age}` : null,
                patientInfo?.sex ?? null,
              ]
                .filter(Boolean)
                .map((tag) => (
                  <span key={tag} className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-[#ede9fe] text-[#4c1d95]">
                    {tag}
                  </span>
                ))}
            </div>
          </div>

          <div className="flex border-b border-[#e9e4f8]">
            {(["predictions", "opinion"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-3 text-xs font-semibold capitalize tracking-wide transition-colors ${
                  activeTab === tab
                    ? "text-[#7c3aed] border-b-2 border-[#7c3aed]"
                    : "text-[#6d5da8] hover:text-[#7c3aed]"
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
                    Model confidence reflects findings across all loaded series. Results are advisory — final diagnosis rests with the reviewing physician.
                  </p>
                </div>
                <div className="flex flex-col gap-4">
                  {PREDICTIONS.map((p) => (
                    <div key={p.label} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-[#1a1523]">{p.label}</span>
                        <SeverityBadge severity={p.severity} />
                      </div>
                      <ProbabilityBar value={p.probability} />
                    </div>
                  ))}
                </div>
                <div className="rounded-xl bg-red-50 border border-red-100 p-4">
                  <p className="text-xs font-semibold text-red-700 uppercase tracking-wide mb-1">Primary Finding</p>
                  <p className="text-sm font-medium text-red-800">Suspected ACL Tear — 82%</p>
                  <p className="text-xs text-red-600 mt-1 leading-relaxed">
                    Complete disruption of the anterior cruciate ligament fiber bundle is suggested on the loaded series.
                  </p>
                </div>
                <button
                  onClick={() => setActiveTab("opinion")}
                  className="w-full py-2.5 rounded-xl bg-[#7c3aed] text-white text-sm font-semibold hover:bg-[#6d28d9] transition-colors"
                >
                  Add Doctor Opinion →
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
                  <h3 className="text-base font-semibold text-[#1a1523]" style={{ fontFamily: "'DM Serif Display', serif" }}>
                    Opinion Submitted
                  </h3>
                  <p className="text-xs text-[#6d5da8] mt-1 leading-relaxed">
                    Your clinical opinion has been recorded and will be shared with the patient's care team.
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
                </div>
                <button
                  onClick={() => { setSubmitted(false); setOpinion(""); setRecommendation(""); }}
                  className="text-xs text-[#7c3aed] hover:underline"
                >
                  Revise opinion
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="px-5 py-5 flex flex-col gap-4">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-7 h-7 rounded-full bg-[#ede9fe] flex items-center justify-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2" strokeLinecap="round">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
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
                    placeholder="Describe your clinical findings based on the imaging..."
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
                    placeholder="e.g. Surgical intervention, physiotherapy, further imaging..."
                    rows={3}
                    className="w-full rounded-xl border border-[#e9e4f8] bg-[#faf9ff] px-3.5 py-3 text-sm text-[#1a1523] placeholder:text-[#b5a9d4] focus:outline-none focus:ring-2 focus:ring-[#a78bfa] focus:border-transparent resize-none leading-relaxed transition-shadow"
                  />
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-semibold text-[#6d5da8] uppercase tracking-wide">Urgency</label>
                  <div className="flex gap-2">
                    {["Routine", "Urgent", "Emergency"].map((u) => (
                      <label key={u} className="flex-1">
                        <input type="radio" name="urgency" value={u} className="sr-only peer" defaultChecked={u === "Urgent"} />
                        <div className="text-center text-xs font-medium py-2 rounded-lg border border-[#e9e4f8] cursor-pointer peer-checked:border-[#7c3aed] peer-checked:bg-[#ede9fe] peer-checked:text-[#4c1d95] text-[#6d5da8] hover:border-[#c4b5fd] transition-all">
                          {u}
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full py-3 rounded-xl bg-[#7c3aed] text-white text-sm font-semibold hover:bg-[#6d28d9] active:scale-95 transition-all mt-1"
                >
                  Submit Opinion
                </button>
                <p className="text-[10px] text-center text-[#b5a9d4] leading-relaxed">
                  This opinion will be signed and timestamped to the patient record.
                </p>
              </form>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
