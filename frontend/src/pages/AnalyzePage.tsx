import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { analyzeImage, analyzeBatch, deleteAnalysis, clearAnalyses, getUploadUrl, type AnalysisResult } from '../api';

export default function AnalyzePage() {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mode, setMode] = useState<'single' | 'batch'>('single');

  const removeResult = async (result: AnalysisResult) => {
    if (!window.confirm(`Delete the analysis for “${result.original_filename}”?`)) return;
    try {
      await deleteAnalysis(result.id);
      setResults((items) => items.filter((item) => item.id !== result.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete analysis.');
    }
  };

  const clearHistory = async () => {
    if (!window.confirm('Delete all analysis history and uploaded analysis images?')) return;
    try {
      await clearAnalyses();
      setResults([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not clear analysis history.');
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    setLoading(true);
    setError('');

    try {
      if (mode === 'batch' && acceptedFiles.length > 1) {
        const res = await analyzeBatch(acceptedFiles);
        setResults((prev) => [...res.results, ...prev]);
        if (res.total_errors > 0) {
          setError(`${res.total_errors} file(s) could not be processed.`);
        }
      } else {
        // Single file or batch mode with 1 file
        for (const file of acceptedFiles) {
          const res = await analyzeImage(file);
          setResults((prev) => [res, ...prev]);
        }
      }
    } catch (e: any) {
      setError(e.message || 'Analysis failed.');
    } finally {
      setLoading(false);
    }
  }, [mode]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/tiff': ['.tiff', '.tif'],
      'image/bmp': ['.bmp'],
    },
    maxSize: 50 * 1024 * 1024,
    multiple: mode === 'batch',
  });

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analyze Images</h1>
        <p className="page-subtitle">
          Upload microscopy or laboratory images for preliminary microplastic screening.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="tab-nav">
        <button
          className={`tab-item ${mode === 'single' ? 'active' : ''}`}
          onClick={() => setMode('single')}
        >
          Single Image
        </button>
        <button
          className={`tab-item ${mode === 'batch' ? 'active' : ''}`}
          onClick={() => setMode('batch')}
        >
          Batch Upload
        </button>
      </div>

      {/* Upload zone */}
      <div className="card mb-lg">
        <div className="card-body">
          <div
            {...getRootProps()}
            className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
          >
            <input {...getInputProps()} id="image-upload" />
            <div className="upload-zone-icon">
              {loading ? '⏳' : isDragActive ? '📥' : '📤'}
            </div>
            <div className="upload-zone-text">
              {loading
                ? 'Processing…'
                : isDragActive
                ? 'Drop images here'
                : mode === 'batch'
                ? 'Drag & drop images or click to select (multiple)'
                : 'Drag & drop an image or click to select'}
            </div>
            <div className="upload-zone-hint">
              Supported: JPG, JPEG, PNG, TIFF, BMP · Max 50 MB per file
            </div>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-error mb-lg">
          <span className="alert-icon">❌</span>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <span>Analyzing image…</span>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="result-toolbar">
          <div>
            <h2 className="section-title">Analysis results</h2>
            <span className="text-sm text-muted">{results.length} result{results.length === 1 ? '' : 's'} in this session</span>
          </div>
          <button className="btn btn-quiet btn-sm" onClick={clearHistory}>
            Clear analysis history
          </button>
        </div>
      )}
      {results.map((r) => (
        <div key={r.id} className="card result-card mb-lg">
          <div className="card-body">
            <div className="result-grid">
              {/* Image preview */}
              <div className="result-image-container">
                <img
                  src={getUploadUrl(r.storage_path)}
                  alt={r.original_filename}
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>

              {/* Details */}
              <div className="result-details">
                <div className="result-meta-header">
                  <div className="text-xs text-muted">
                    {r.original_filename}
                    {r.image_width && r.image_height && (
                      <> · {r.image_width}×{r.image_height}px</>
                    )}
                    {r.created_at && (
                      <> · {new Date(r.created_at).toLocaleString()}</>
                    )}
                  </div>
                  <button className="btn btn-quiet btn-sm" onClick={() => removeResult(r)}>
                    Delete
                  </button>
                </div>

                {/* Prediction */}
                <div className="result-prediction">
                  <span className={`prediction-class ${r.predicted_class || (r.status === 'uncertain' ? 'uncertain' : '')}`}>
                    {r.status === 'out_of_domain'
                      ? 'Outside training image scope'
                      : r.status === 'model_unavailable'
                      ? 'Model unavailable'
                      : r.status === 'model_requires_validation'
                      ? 'Model needs validation'
                      : r.status === 'uncertain'
                      ? 'Uncertain · review required'
                      : r.predicted_class?.replace('_', ' ') || 'Unknown'}
                  </span>
                  {r.status === 'out_of_domain' && <span className="badge badge-warning">no class predicted</span>}
                  {r.status === 'model_unavailable' && <span className="badge badge-warning">no prediction</span>}
                  {r.status === 'model_requires_validation' && <span className="badge badge-warning">prediction withheld</span>}
                  {r.status === 'uncertain' && <span className="badge badge-warning">label withheld</span>}
                  {r.status === 'error' && <span className="badge badge-error">error</span>}
                </div>

                {/* Confidence */}
                {r.confidence !== null && (
                  <div className="confidence-bar-container">
                    <div className="confidence-bar-label">
                      <span>Model score</span>
                      <span>{(r.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className={`confidence-bar-fill ${r.status === 'uncertain' || r.confidence < 0.65 ? 'low' : ''}`}
                        style={{ width: `${r.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Model info */}
                <div className="text-xs text-muted">
                  Model: {r.model_name || '—'} · Version: {r.model_version || '—'}
                </div>

                {r.warnings?.length > 0 && (
                  <div className="alert alert-warning">
                    <span className="alert-icon">⚠️</span>
                    <ul className="result-warnings">
                      {r.warnings.map((warning, index) => <li key={`${r.id}-warning-${index}`}>{warning}</li>)}
                    </ul>
                  </div>
                )}

                {/* Detections */}
                {r.detection_count !== null && r.detection_count > 0 && (
                  <div className="badge badge-info">
                    {r.detection_count} particle(s) detected
                  </div>
                )}

                {/* Disclaimers */}
                <div className="alert alert-info" style={{ marginTop: '8px' }}>
                  <span className="alert-icon">ℹ️</span>
                  <span className="text-xs">{r.confidence_note}</span>
                </div>

                <div className="alert alert-warning" style={{ marginTop: '4px' }}>
                  <span className="alert-icon">⚠️</span>
                  <span className="text-xs">{r.screening_disclaimer}</span>
                </div>

              </div>
            </div>
          </div>
        </div>
      ))}

      {/* Empty state */}
      {!loading && results.length === 0 && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">🔬</div>
            <div className="empty-state-text">No results yet</div>
            <div className="empty-state-hint">
              Upload an image above to begin analysis
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
