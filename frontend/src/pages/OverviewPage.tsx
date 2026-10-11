import { useEffect, useState } from 'react';
import { getStatus, getAnalyses, deleteAnalysis, clearAnalyses, type AppStatus, type AnalysisResult, getUploadUrl } from '../api';
import { Link } from 'react-router-dom';

export default function OverviewPage() {
  const [status, setStatus] = useState<AppStatus | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [historyBusy, setHistoryBusy] = useState(false);

  useEffect(() => {
    Promise.all([
      getStatus(),
      getAnalyses(1, 5),
    ])
      .then(([s, a]) => {
        setStatus(s);
        setRecentAnalyses(a.items);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const removeOne = async (analysis: AnalysisResult) => {
    if (!window.confirm(`Delete the analysis for “${analysis.original_filename}”?`)) return;
    setHistoryBusy(true);
    setError('');
    try {
      await deleteAnalysis(analysis.id);
      setRecentAnalyses((items) => items.filter((item) => item.id !== analysis.id));
      setStatus((current) => current ? { ...current, total_analyses: Math.max(0, current.total_analyses - 1) } : current);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete analysis.');
    } finally {
      setHistoryBusy(false);
    }
  };

  const clearHistory = async () => {
    if (!window.confirm('Delete all analysis history and uploaded analysis images?')) return;
    setHistoryBusy(true);
    setError('');
    try {
      await clearAnalyses();
      setRecentAnalyses([]);
      setStatus((current) => current ? { ...current, total_analyses: 0 } : current);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not clear analysis history.');
    } finally {
      setHistoryBusy(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <span>Loading overview…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-error">
        <span className="alert-icon">❌</span>
        <div>
          <strong>Connection Error</strong>
          <p>Could not connect to backend: {error}</p>
          <p className="text-sm mt-md">Make sure the backend server is running on port 8000.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Overview</h1>
        <p className="page-subtitle">
          Image-based machine learning for preliminary microplastic screening
        </p>
      </div>

      {/* Screening disclaimer */}
      <div className="alert alert-info mb-lg">
        <span className="alert-icon">ℹ️</span>
        <div>
          Image classification provides <strong>preliminary screening only</strong>.
          Confirmed polymer identification requires FTIR, Raman spectroscopy,
          or other chemical/spectroscopic evidence. Do not treat predictions as
          definitive chemical analysis.
        </div>
      </div>

      {!status?.model_loaded && (
        <div className="alert alert-demo mb-lg">
          <span className="alert-icon">⚡</span>
          <div>
            <strong>No trained classifier is loaded.</strong> Analysis is paused until a model trained on a validated microscopy dataset is configured.
            See <Link to="/dataset">Dataset readiness</Link> and <Link to="/about">limitations</Link>.
          </div>
        </div>
      )}

      {status?.model_loaded && status.model_ready === false && (
        <div className="alert alert-demo mb-lg">
          <span className="alert-icon">⚠️</span>
          <div>
            <strong>Classifier predictions are paused.</strong> The loaded model has not met the 75% held-out macro-F1 readiness threshold. Upload results will be marked as needing validation until it is evaluated on stronger, representative microscopy data.
          </div>
        </div>
      )}

      {status?.model_scope && (
        <div className="alert alert-info mb-lg">
          <span className="alert-icon">🔬</span>
          <div><strong>What this model is trained for:</strong> {status.model_scope}</div>
        </div>
      )}

      {/* Stats */}
      <div className="stats-grid">
        <div className="card stat-card stat-card--primary">
          <div className="stat-label">Total Analyses</div>
          <div className="stat-value">{status?.total_analyses ?? 0}</div>
          <div className="stat-note">Images processed</div>
        </div>
        <div className="card stat-card stat-card--info">
          <div className="stat-label">Model Runs</div>
          <div className="stat-value">{status?.total_runs ?? 0}</div>
          <div className="stat-note">Training sessions</div>
        </div>
        <div className="card stat-card stat-card--warning">
          <div className="stat-label">Active Model</div>
          <div className="stat-value" style={{ fontSize: '1.25rem' }}>
            {status?.model_loaded ? status.active_model || 'Loaded' : 'None'}
          </div>
          <div className="stat-note">
            {!status?.model_loaded ? 'No model configured' : status.model_ready === false ? 'Predictions withheld pending validation' : 'Ready for inference'}
          </div>
        </div>
        <div className="card stat-card stat-card--demo">
          <div className="stat-label">Mode</div>
          <div className="stat-value" style={{ fontSize: '1.25rem' }}>
            {status?.demo_mode ? 'Demo' : status?.model_ready === false ? 'Needs validation' : 'Production'}
          </div>
          <div className="stat-note">
            {status?.demo_mode ? 'Illustrative predictions' : status?.model_ready === false ? 'Classifier is paused' : 'Trained model active'}
          </div>
        </div>
      </div>

      {/* Workflow */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Screening Workflow</h2>
        </div>
        <div className="card-body">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px' }}>
            {[
              { step: '1', title: 'Upload Image', desc: 'Microscopy or laboratory image', icon: '📤' },
              { step: '2', title: 'Preprocessing', desc: 'Resize, normalize, validate', icon: '🔧' },
              { step: '3', title: 'Classification', desc: 'CNN / transfer-learning model', icon: '🧠' },
              { step: '4', title: 'Review Results', desc: 'Class, confidence, limitations', icon: '📋' },
            ].map((s) => (
              <div key={s.step} style={{
                textAlign: 'center',
                padding: '20px 12px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--color-neutral-50)',
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>{s.icon}</div>
                <div style={{ fontWeight: 700, fontSize: 'var(--font-size-sm)', marginBottom: '4px' }}>
                  Step {s.step}: {s.title}
                </div>
                <div className="text-xs text-muted">{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Analyses</h2>
          <div className="header-actions">
            {recentAnalyses.length > 0 && (
              <button className="btn btn-quiet btn-sm" onClick={clearHistory} disabled={historyBusy}>
                Clear history
              </button>
            )}
            <Link to="/analyze" className="btn btn-primary btn-sm">
              New Analysis
            </Link>
          </div>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          {recentAnalyses.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔬</div>
              <div className="empty-state-text">No analyses yet</div>
              <div className="empty-state-hint">
                Upload an image to get started
              </div>
            </div>
          ) : (
            recentAnalyses.map((a) => (
              <div key={a.id} className="history-item">
                <img
                  className="history-thumb"
                  src={getUploadUrl(a.storage_path)}
                  alt={a.original_filename}
                  onError={(e) => {
                    (e.target as HTMLImageElement).src =
                      'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect fill="%23e2e7eb" width="48" height="48" rx="4"/><text x="24" y="28" text-anchor="middle" fill="%239aa5b0" font-size="16">🔬</text></svg>';
                  }}
                />
                <div className="history-info">
                  <div className="history-filename">{a.original_filename}</div>
                  <div className="history-meta">
                    {a.predicted_class && (
                      <span className={`badge ${a.predicted_class === 'microplastic' ? 'badge-error' : 'badge-success'}`}>
                        {a.predicted_class.replace('_', ' ')}
                      </span>
                    )}
                    {a.status === 'uncertain' && <span className="badge badge-warning">uncertain</span>}
                    {' '}
                    {a.confidence !== null && (
                      <span className="text-xs text-muted">
                        {(a.confidence * 100).toFixed(1)}% confidence
                      </span>
                    )}
                    {a.is_demo && <span className="badge badge-demo" style={{ marginLeft: 4 }}>demo</span>}
                  </div>
                </div>
                <div className="text-xs text-muted">
                  {a.created_at ? new Date(a.created_at).toLocaleString() : '—'}
                </div>
                <button
                  className="btn btn-icon btn-danger-quiet"
                  onClick={() => removeOne(a)}
                  disabled={historyBusy}
                  aria-label={`Delete analysis for ${a.original_filename}`}
                  title="Delete analysis"
                >
                  <span aria-hidden="true">×</span>
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
