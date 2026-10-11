const API_BASE = import.meta.env.VITE_API_URL || '';

export interface AnalysisResult {
  id: string;
  created_at: string | null;
  original_filename: string;
  storage_path: string;
  file_type: string;
  image_width: number | null;
  image_height: number | null;
  predicted_class: string | null;
  confidence: number | null;
  model_name: string | null;
  model_version: string | null;
  status: string;
  error_message: string | null;
  is_demo: boolean;
  warnings: string[];
  preprocessing_config: Record<string, unknown> | null;
  detections: Array<Record<string, unknown>> | null;
  detection_count: number | null;
  confidence_note: string;
  screening_disclaimer: string;
}

export interface AnalysisList {
  items: AnalysisResult[];
  total: number;
  page: number;
  per_page: number;
}

export interface BatchResult {
  results: AnalysisResult[];
  total_processed: number;
  total_errors: number;
}

export interface AppStatus {
  app_name: string;
  demo_mode: boolean;
  model_loaded: boolean;
  model_ready?: boolean;
  model_scope?: string | null;
  active_model: string | null;
  yolo_available: boolean;
  confidence_threshold: number;
  total_analyses: number;
  total_runs: number;
}

export interface DatasetInfo {
  id: string;
  created_at: string | null;
  name: string;
  description: string | null;
  source: string | null;
  license_notes: string | null;
  base_path: string;
  total_images: number;
  class_counts: Record<string, number> | null;
  split_counts: Record<string, number> | null;
  validation_warnings: string[] | null;
  label_provenance: string | null;
  is_valid: boolean;
  training_ready?: boolean;
}

export interface TrainingRun {
  id: string;
  created_at: string | null;
  updated_at: string | null;
  run_name: string;
  status: string;
  model_architecture: string;
  dataset_manifest: string | null;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size: number;
  checkpoint_path: string | null;
  best_val_accuracy: number | null;
  training_history: Array<Record<string, number>> | null;
  class_mapping: Record<string, string> | null;
  train_count: number | null;
  val_count: number | null;
  test_count: number | null;
  split_method?: string | null;
  test_metrics?: Record<string, unknown> | null;
  error_message: string | null;
  progress: number;
}

export interface EvaluationResult {
  id: string;
  created_at: string | null;
  run_id: string;
  test_split_name: string;
  test_count: number | null;
  class_counts: Record<string, number> | null;
  accuracy: number | null;
  precision_per_class: Record<string, number> | null;
  recall_per_class: Record<string, number> | null;
  f1_per_class: Record<string, number> | null;
  precision_macro: number | null;
  recall_macro: number | null;
  f1_macro: number | null;
  confusion_matrix: number[][] | null;
  class_order: string[] | null;
  model_checkpoint: string | null;
  status: string;
  error_message: string | null;
  evaluation_note: string;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...options?.headers,
    },
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `API Error: ${resp.status}`);
  }
  return resp.json();
}

// ── System ────────────────────────────────────────────────────

export const getHealth = () => apiFetch<{ status: string; version: string }>('/api/health');
export const getStatus = () => apiFetch<AppStatus>('/api/status');

// ── Analysis ──────────────────────────────────────────────────

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);
  return apiFetch<AnalysisResult>('/api/analyze', {
    method: 'POST',
    body: formData,
  });
}

export async function analyzeBatch(files: File[]): Promise<BatchResult> {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  return apiFetch<BatchResult>('/api/analyze/batch', {
    method: 'POST',
    body: formData,
  });
}

export const getAnalyses = (page = 1, perPage = 20) =>
  apiFetch<AnalysisList>(`/api/analyses?page=${page}&per_page=${perPage}`);

export const deleteAnalysis = (id: string) =>
  apiFetch<{ deleted: number }>(`/api/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' });

export const clearAnalyses = () =>
  apiFetch<{ deleted: number }>('/api/analyses', { method: 'DELETE' });

export const getAnalysis = (id: string) =>
  apiFetch<AnalysisResult>(`/api/analyses/${id}`);

// ── Dataset ───────────────────────────────────────────────────

export const getDatasets = () => apiFetch<DatasetInfo[]>('/api/datasets');

export const validateDataset = (basePath: string, name?: string) =>
  apiFetch<DatasetInfo>('/api/datasets/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base_path: basePath, name }),
  });

// ── Model Runs ────────────────────────────────────────────────

export const getRuns = () => apiFetch<TrainingRun[]>('/api/runs');
export const getRun = (id: string) => apiFetch<TrainingRun>(`/api/runs/${id}`);

export async function createRun(data: {
  run_name: string;
  model_architecture: string;
  dataset_path: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  image_size: number;
  seed: number;
  use_class_weights: boolean;
}): Promise<TrainingRun> {
  return apiFetch<TrainingRun>('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

// ── Evaluation ────────────────────────────────────────────────

export const getEvaluations = (runId: string) =>
  apiFetch<EvaluationResult[]>(`/api/evaluations/${runId}`);

export const runEvaluation = (runId: string, testSplitName = 'test') =>
  apiFetch<EvaluationResult>('/api/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, test_split_name: testSplitName }),
  });

// ── Models ────────────────────────────────────────────────────

export const getModels = () =>
  apiFetch<{
    available_architectures: string[];
    active_model: { loaded: boolean; name: string | null; version: string | null };
  }>('/api/models');

// ── Uploads ───────────────────────────────────────────────────

export const getUploadUrl = (storagePath: string) => {
  const filename = storagePath.split(/[\\/]/).pop() || '';
  return `${API_BASE}/api/uploads/${filename}`;
};
