import { useEffect, useState } from 'react';
import { getRuns, createRun, getModels, type TrainingRun } from '../api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

export default function RunsPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [architectures, setArchitectures] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);

  // Form state
  const [form, setForm] = useState({
    run_name: '',
    model_architecture: 'mobilenet_v3_small',
    dataset_path: '',
    epochs: 10,
    batch_size: 32,
    learning_rate: 0.0001,
    image_size: 224,
    seed: 42,
    use_class_weights: false,
  });

  useEffect(() => {
    Promise.all([getRuns(), getModels()])
      .then(([r, m]) => {
        setRuns(r);
        setArchitectures(m.available_architectures);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleCreate = async () => {
    if (!form.run_name || !form.dataset_path) return;
    setCreating(true);
    setError('');
    try {
      const run = await createRun(form);
      setRuns((prev) => [run, ...prev]);
      setShowCreate(false);
      setForm({ ...form, run_name: '', dataset_path: '' });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCreating(false);
    }
  };

  const refreshRuns = () => {
    getRuns().then(setRuns).catch(() => {});
  };

  const statusColor: Record<string, string> = {
    completed: 'badge-success',
    training: 'badge-info',
    pending: 'badge-pending',
    failed: 'badge-error',
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <span>Loading model runs…</span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="page-title">Model Runs</h1>
            <p className="page-subtitle">
              Train classifiers on registered datasets and track run history.
            </p>
          </div>
          <div className="flex gap-sm">
            <button className="btn btn-secondary btn-sm" onClick={refreshRuns}>
              Refresh
            </button>
            <button
              className="btn btn-primary"
              onClick={() => setShowCreate(!showCreate)}
            >
              {showCreate ? 'Cancel' : 'New Training Run'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-lg">
          <span className="alert-icon">❌</span>
          {error}
        </div>
      )}

      {/* Create form */}
      {showCreate && (
        <div className="card mb-lg" style={{ animation: 'slideUp 0.3s ease-out' }}>
          <div className="card-header">
            <h2 className="card-title">New Training Run</h2>
          </div>
          <div className="card-body">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div className="form-group">
                <label htmlFor="run-name" className="form-label">Run Name *</label>
                <input
                  id="run-name"
                  className="form-input"
                  placeholder="e.g. ResNet-18 Lab Samples v1"
                  value={form.run_name}
                  onChange={(e) => setForm({ ...form, run_name: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="architecture" className="form-label">Architecture</label>
                <select
                  id="architecture"
                  className="form-input form-select"
                  value={form.model_architecture}
                  onChange={(e) => setForm({ ...form, model_architecture: e.target.value })}
                >
                  {architectures.map((a) => (
                    <option key={a} value={a}>{a}</option>
                  ))}
                </select>
              </div>
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label htmlFor="dataset-path" className="form-label">Dataset Path *</label>
                <input
                  id="dataset-path"
                  className="form-input"
                  placeholder="e.g. data/datasets/my_dataset"
                  value={form.dataset_path}
                  onChange={(e) => setForm({ ...form, dataset_path: e.target.value })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="epochs" className="form-label">Epochs</label>
                <input
                  id="epochs"
                  type="number"
                  className="form-input"
                  value={form.epochs}
                  onChange={(e) => setForm({ ...form, epochs: parseInt(e.target.value) || 10 })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="batch-size" className="form-label">Batch Size</label>
                <input
                  id="batch-size"
                  type="number"
                  className="form-input"
                  value={form.batch_size}
                  onChange={(e) => setForm({ ...form, batch_size: parseInt(e.target.value) || 16 })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="lr" className="form-label">Learning Rate</label>
                <input
                  id="lr"
                  type="number"
                  step="0.0001"
                  className="form-input"
                  value={form.learning_rate}
                  onChange={(e) => setForm({ ...form, learning_rate: parseFloat(e.target.value) || 0.001 })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="img-size" className="form-label">Image Size</label>
                <input
                  id="img-size"
                  type="number"
                  className="form-input"
                  value={form.image_size}
                  onChange={(e) => setForm({ ...form, image_size: parseInt(e.target.value) || 224 })}
                />
              </div>
              <div className="form-group">
                <label htmlFor="seed" className="form-label">Random Seed</label>
                <input
                  id="seed"
                  type="number"
                  className="form-input"
                  value={form.seed}
                  onChange={(e) => setForm({ ...form, seed: parseInt(e.target.value) || 42 })}
                />
              </div>
              <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '8px', paddingTop: '24px' }}>
                <input
                  id="class-weights"
                  type="checkbox"
                  checked={form.use_class_weights}
                  onChange={(e) => setForm({ ...form, use_class_weights: e.target.checked })}
                />
                <label htmlFor="class-weights" className="text-sm">Use class weights (for imbalanced data)</label>
              </div>
            </div>

            <button
              className="btn btn-primary btn-lg mt-lg"
              onClick={handleCreate}
              disabled={creating || !form.run_name || !form.dataset_path}
            >
              {creating ? 'Starting…' : 'Start Training'}
            </button>
          </div>
        </div>
      )}

      {/* Runs list */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Training History</h2>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          {runs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">⚙️</div>
              <div className="empty-state-text">No training runs</div>
              <div className="empty-state-hint">
                Click "New Training Run" to train your first model.
              </div>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run Name</th>
                  <th>Architecture</th>
                  <th>Status</th>
                  <th>Best Val Acc</th>
                  <th>Splits</th>
                  <th>Config</th>
                  <th>Date</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <>
                    <tr key={r.id}>
                      <td className="font-bold">{r.run_name}</td>
                      <td>{r.model_architecture}</td>
                      <td>
                        <span className={`badge ${statusColor[r.status] || 'badge-pending'}`}>
                          {r.status}
                        </span>
                      </td>
                      <td>
                        {r.best_val_accuracy !== null
                          ? `${(r.best_val_accuracy * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="text-xs">
                        {r.train_count !== null && (
                          <span>
                            Train: {r.train_count} / Val: {r.val_count} / Test: {r.test_count}
                            {r.test_metrics && <> · Test accuracy: {Number(r.test_metrics.accuracy ?? 0).toFixed(3)} · Macro F1: {Number(r.test_metrics.macro_f1 ?? 0).toFixed(3)}</>}
                          </span>
                        )}
                      </td>
                      <td className="text-xs text-muted">
                        E:{r.epochs} B:{r.batch_size} LR:{r.learning_rate}
                      </td>
                      <td className="text-xs text-muted">
                        {r.created_at ? new Date(r.created_at).toLocaleDateString() : '—'}
                      </td>
                      <td>
                        {r.training_history && r.training_history.length > 0 && (
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={() => setExpandedRunId(expandedRunId === r.id ? null : r.id)}
                          >
                            {expandedRunId === r.id ? 'Hide' : 'Chart'}
                          </button>
                        )}
                      </td>
                    </tr>
                    {expandedRunId === r.id && r.training_history && (
                      <tr key={`${r.id}-chart`}>
                        <td colSpan={8} style={{ padding: '16px' }}>
                          <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={r.training_history}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7eb" />
                              <XAxis
                                dataKey="epoch"
                                tick={{ fontSize: 12, fill: '#6b7a87' }}
                                label={{ value: 'Epoch', position: 'insideBottomRight', fill: '#6b7a87' }}
                              />
                              <YAxis tick={{ fontSize: 12, fill: '#6b7a87' }} />
                              <Tooltip />
                              <Legend />
                              <Line type="monotone" dataKey="train_loss" stroke="#ef4444" name="Train Loss" strokeWidth={2} dot={false} />
                              <Line type="monotone" dataKey="val_loss" stroke="#3b82f6" name="Val Loss" strokeWidth={2} dot={false} />
                              <Line type="monotone" dataKey="val_accuracy" stroke="#0d9668" name="Val Accuracy" strokeWidth={2} dot={false} />
                            </LineChart>
                          </ResponsiveContainer>
                          {r.error_message && (
                            <div className="alert alert-error mt-md">
                              <span className="alert-icon">❌</span>
                              {r.error_message}
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
