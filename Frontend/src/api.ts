// ─── API Integration Layer ────────────────────────────────────────────────────
//
// Every function below is a clearly marked integration point.
// Replace the mock returns with real fetch() calls once the backend is ready.
// Set VITE_API_BASE_URL in your .env file (e.g. http://localhost:8000).
//
// ─────────────────────────────────────────────────────────────────────────────

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ─── Shared types ─────────────────────────────────────────────────────────────

export type StudyStatus = "pending" | "urgent" | "reviewed";

export type Study = {
  id: string;
  patientName: string;
  patientAge: string;
  patientSex: string;
  studyDate: string;
  accessionNumber: string;
  bodyPart: string;
  seriesCount: number;
  status: StudyStatus;
  primaryFinding: string | null;
  predictions: Prediction[] | null;
  report: Report | null;
};

export type Label = {
  id: string;
  name: string;
  description: string;
  severity: "High" | "Moderate" | "Low";
};

export type Prediction = {
  labelId: string;
  label: string;
  probability: number;
  severity: "High" | "Moderate" | "Low";
};

export type Report = {
  clinicalImpression: string;
  recommendation: string;
  urgency: "Routine" | "Urgent" | "Emergency";
  doctorName: string;
  submittedAt: string;
};

// ─── Mock data (remove once real API is connected) ────────────────────────────

// const MOCK_LABELS: Label[] = [
//   { id: "acl", name: "ACL Tear", description: "Rupture of the anterior cruciate ligament", severity: "High" },
//   { id: "meniscal", name: "Meniscal Tear", description: "Tear in the meniscus cartilage", severity: "Moderate" },
//   { id: "cartilage", name: "Cartilage Damage", description: "Degradation of articular cartilage", severity: "Moderate" },
//   { id: "edema", name: "Bone Edema", description: "Fluid accumulation in bone marrow", severity: "Low" },
//   { id: "pcl", name: "PCL Tear", description: "Posterior cruciate ligament injury", severity: "High" },
//   { id: "bakers", name: "Baker's Cyst", description: "Fluid-filled cyst behind the knee", severity: "Low" },
// ];

// const MOCK_PREDICTIONS: Prediction[] = [
//   { labelId: "acl", label: "ACL Tear", probability: 0.82, severity: "High" },
//   { labelId: "meniscal", label: "Meniscal Tear", probability: 0.61, severity: "Moderate" },
//   { labelId: "cartilage", label: "Cartilage Damage", probability: 0.38, severity: "Low" },
//   { labelId: "edema", label: "Bone Edema", probability: 0.27, severity: "Low" },
// ];

// const MOCK_STUDIES: Study[] = [
//   {
//     id: "study-001",
//     patientName: "Sarah Mitchell",
//     patientAge: "34Y",
//     patientSex: "F",
//     studyDate: "2026-08-24",
//     accessionNumber: "MRI-2026-08241",
//     bodyPart: "Right Knee",
//     seriesCount: 4,
//     status: "pending",
//     primaryFinding: "Suspected ACL Tear (82%)",
//     predictions: MOCK_PREDICTIONS,
//     report: null,
//   },
//   {
//     id: "study-002",
//     patientName: "James Cooper",
//     patientAge: "41Y",
//     patientSex: "M",
//     studyDate: "2026-08-22",
//     accessionNumber: "MRI-2026-08221",
//     bodyPart: "Left Knee",
//     seriesCount: 3,
//     status: "urgent",
//     primaryFinding: "Meniscal Tear (74%)",
//     predictions: [
//       { labelId: "meniscal", label: "Meniscal Tear", probability: 0.74, severity: "Moderate" },
//       { labelId: "edema", label: "Bone Edema", probability: 0.52, severity: "Low" },
//     ],
//     report: null,
//   },
//   {
//     id: "study-003",
//     patientName: "Amara Osei",
//     patientAge: "28Y",
//     patientSex: "F",
//     studyDate: "2026-08-19",
//     accessionNumber: "MRI-2026-08191",
//     bodyPart: "Right Knee",
//     seriesCount: 6,
//     status: "reviewed",
//     primaryFinding: "No significant finding",
//     predictions: [
//       { labelId: "cartilage", label: "Cartilage Damage", probability: 0.18, severity: "Low" },
//     ],
//     report: {
//       clinicalImpression: "No acute ligamentous injury. Mild chondral thinning noted.",
//       recommendation: "Conservative management. Physiotherapy referral.",
//       urgency: "Routine",
//       doctorName: "Dr. James Okafor",
//       submittedAt: "2026-08-20T09:14:00Z",
//     },
//   },
//   {
//     id: "study-004",
//     patientName: "Liu Wei",
//     patientAge: "55Y",
//     patientSex: "M",
//     studyDate: "2026-08-15",
//     accessionNumber: "MRI-2026-08151",
//     bodyPart: "Left Knee",
//     seriesCount: 4,
//     status: "reviewed",
//     primaryFinding: "Cartilage Damage (61%)",
//     predictions: [
//       { labelId: "cartilage", label: "Cartilage Damage", probability: 0.61, severity: "Moderate" },
//       { labelId: "edema", label: "Bone Edema", probability: 0.44, severity: "Low" },
//     ],
//     report: {
//       clinicalImpression: "Grade III chondral lesion on medial femoral condyle.",
//       recommendation: "Orthopaedic consultation for possible microfracture procedure.",
//       urgency: "Urgent",
//       doctorName: "Dr. James Okafor",
//       submittedAt: "2026-08-16T14:32:00Z",
//     },
//   },
// ];

// ─── API methods ──────────────────────────────────────────────────────────────

export const api = {
  /**
   * GET /studies
   * Returns all studies stored in the database.
   */
  async getStudies(): Promise<Study[]> {
    const res = await fetch(`${API_BASE_URL}/studies`);
    if (!res.ok) throw new Error("Failed to fetch studies");
    return res.json();
  },

  /**
   * GET /labels
   * Returns the list of injury labels the model can predict.
   * Load once on app start and cache.
   */
  async getLabels(): Promise<Label[]> {
    const res = await fetch(`${API_BASE_URL}/labels`);
    if (!res.ok) throw new Error("Failed to fetch labels");
    return res.json();
  },

  /**
   * POST /predict
   * Sends one or more DICOM files to the model and returns predictions.
   * Files are sent as multipart/form-data under the key "files".
   */
  async predict(files: File[]): Promise<Prediction[]> {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: form });
    if (!res.ok) throw new Error("Prediction failed");
    return res.json();
  },

  /**
   * POST /studies
   * Creates a new study record from uploaded DICOM files and patient metadata.
   * Returns the created study with a server-assigned ID.
   */
  async createStudy(
    files: File[],
    patientName: string,
    predictions: Prediction[]
  ): Promise<Study> {
    // TODO:
    // const form = new FormData();
    // files.forEach((f) => form.append("files", f));
    // form.append("patientName", patientName);
    // const res = await fetch(`${API_BASE_URL}/studies`, { method: "POST", body: form });
    // if (!res.ok) throw new Error("Failed to create study");
    // return res.json();
    const primary = predictions[0] ?? null;
    return {
      id: `study-${Date.now()}`,
      patientName: patientName || "Unknown Patient",
      patientAge: "",
      patientSex: "",
      studyDate: new Date().toISOString().slice(0, 10),
      accessionNumber: `MRI-${Date.now()}`,
      bodyPart: "Knee",
      seriesCount: files.length,
      status: "pending",
      primaryFinding: primary ? `${primary.label} (${Math.round(primary.probability * 100)}%)` : null,
      predictions,
      report: null,
    };
  },

  /**
   * POST /studies/:id/report
   * Saves a doctor's clinical report for a study.
   */
  async saveReport(studyId: string, report: Report): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/studies/${studyId}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(report),
    });
    if (!res.ok) throw new Error("Failed to save report");
  },
}
