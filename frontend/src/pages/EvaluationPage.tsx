import { useEffect, useState } from 'react';
import { getRuns, getEvaluations, runEvaluation, type TrainingRun, type EvaluationResult } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, Legend,
} from 'recharts';

export default function EvaluationPage() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [evaluations, setEvaluations] = useState<EvaluationResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [evalLoading, setEvalLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getRuns()
      .then((r) => {
        setRuns(r.filter((run) => run.status === 'completed'));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      setEvalLoading(true);
      getEvaluations(selectedRunId)
        .then(setEvaluations)
        .catch((e) => setError(e.message))
        .finally(() => setEvalLoading(false));
    }
  }, [selectedRunId]);

  const handleEvaluate = async () => {
    if (!selectedRunId) return;
    setEvalLoading(true);
    setError('');
    try {
      const result = await runEvaluation(selectedRunId);
      setEvaluations((prev) => [result, ...prev]);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setEvalLoading(false);
    }
  };

  const latestEval = evaluations.length > 0 ? evaluations[0] : null;

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <span>Loading evaluation data…</span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Evaluation</h1>
        <p className="page-subtitle">
          Evaluate trained models on held-out test sets. Metrics are specific to the test
          data used and may not generalise to other environments.
        </p>
      </div>

      {/* Run selector */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Select Training Run</h2>
        </div>
        <div className="card-body">
          {runs.length === 0 ? (
            <div className="alert alert-info">
              <span className="alert-icon">ℹ️</span>
              No completed training runs available. Train a model first from the Model Runs page.
            </div>
          ) : (
            <div className="flex gap-md items-center">
              <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
                <select
                  id="run-select"
                  className="form-input form-select"
                  value={selectedRunId}
                  onChange={(e) => setSelectedRunId(e.target.value)}
                >
                  <option value="">Choose a run…</option>
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.run_name} ({r.model_architecture}) – Val Acc: {r.best_val_accuracy !== null ? (r.best_val_accuracy * 100).toFixed(1) + '%' : 'N/A'}
                    </option>
                  ))}
                </select>
              </div>
              <button
                className="btn btn-primary"
                onClick={handleEvaluate}
                disabled={!selectedRunId || evalLoading}
              >
                {evalLoading ? 'Evaluating…' : 'Run Evaluation'}
              </button>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-lg">
          <span className="alert-icon">❌</span>
          {error}
        </div>
      )}

      {/* No evaluation state */}
      {!latestEval && !evalLoading && (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📈</div>
            <div className="empty-state-text">No evaluation results</div>
            <div className="empty-state-hint">
              {runs.length > 0
                ? 'Select a training run and click "Run Evaluation" to calculate metrics.'
                : 'No evaluation has been performed. Complete a training run first.'}
            </div>
          </div>
        </div>
      )}

      {evalLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <span>Evaluating model on test set…</span>
        </div>
      )}

      {/* Evaluation results */}
      {latestEval && latestEval.status === 'completed' && (
        <>
          {/* Summary metrics */}
          <div className="metrics-grid mb-lg">
            {[
              { label: 'Accuracy', value: latestEval.accuracy },
              { label: 'Precision (Macro)', value: latestEval.precision_macro },
              { label: 'Recall (Macro)', value: latestEval.recall_macro },
              { label: 'F1 Score (Macro)', value: latestEval.f1_macro },
            ].map((m) => (
              <div key={m.label} className="card metric-card">
                <div className="metric-value">
                  {m.value !== null ? (m.value * 100).toFixed(1) + '%' : 'N/A'}
                </div>
                <div className="metric-label">{m.label}</div>
              </div>
            ))}
          </div>

          {/* Per-class metrics chart */}
          {latestEval.f1_per_class && (
            <div className="card mb-lg">
              <div className="card-header">
                <h2 className="card-title">Per-Class Metrics</h2>
              </div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={Object.keys(latestEval.f1_per_class).map((cls) => ({
                      name: cls.replace('_', ' '),
                      Precision: latestEval.precision_per_class?.[cls] ?? 0,
                      Recall: latestEval.recall_per_class?.[cls] ?? 0,
                      'F1 Score': latestEval.f1_per_class?.[cls] ?? 0,
                    }))}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e7eb" />
                    <XAxis dataKey="name" tick={{ fontSize: 13, fill: '#6b7a87' }} />
                    <YAxis
                      domain={[0, 1]}
                      tick={{ fontSize: 12, fill: '#6b7a87' }}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                      label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: '#6b7a87' }}
                    />
                    <Tooltip formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} />
                    <Legend />
                    <Bar dataKey="Precision" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Recall" fill="#0d9668" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="F1 Score" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Confusion Matrix */}
          {latestEval.confusion_matrix && latestEval.class_order && (
            <div className="card mb-lg">
              <div className="card-header">
                <h2 className="card-title">Confusion Matrix</h2>
              </div>
              <div className="card-body">
                <div className="confusion-matrix">
                  <table>
                    <thead>
                      <tr>
                        <th></th>
                        {latestEval.class_order.map((cls) => (
                          <th key={cls}>Predicted: {cls.replace('_', ' ')}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {latestEval.confusion_matrix.map((row, i) => (
                        <tr key={i}>
                          <th>Actual: {latestEval.class_order![i].replace('_', ' ')}</th>
                          {row.map((val, j) => {
                            const isDiag = i === j;
                            return (
                              <td
                                key={j}
                                className={isDiag ? 'cm-cell-high' : val > 0 ? 'cm-cell-low' : ''}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="text-xs text-muted mt-md">
                  Rows = actual class, Columns = predicted class
                </div>
              </div>
            </div>
          )}

          {/* Provenance */}
          <div className="card mb-lg">
            <div className="card-header">
              <h2 className="card-title">Evaluation Provenance</h2>
            </div>
            <div className="card-body">
              <table className="data-table">
                <tbody>
                  <tr><td className="font-bold">Evaluation ID</td><td>{latestEval.id}</td></tr>
                  <tr><td className="font-bold">Run ID</td><td>{latestEval.run_id}</td></tr>
                  <tr><td className="font-bold">Test Split</td><td>{latestEval.test_split_name}</td></tr>
                  <tr><td className="font-bold">Test Set Size</td><td>{latestEval.test_count ?? 'N/A'}</td></tr>
                  <tr><td className="font-bold">Checkpoint</td><td className="text-xs">{latestEval.model_checkpoint || 'N/A'}</td></tr>
                  <tr><td className="font-bold">Date</td><td>{latestEval.created_at ? new Date(latestEval.created_at).toLocaleString() : 'N/A'}</td></tr>
                  {latestEval.class_counts && (
                    <tr>
                      <td className="font-bold">Class Counts (Test)</td>
                      <td>
                        {Object.entries(latestEval.class_counts).map(([cls, cnt]) => (
                          <span key={cls} style={{ marginRight: 12 }}>
                            {cls.replace('_', ' ')}: {cnt}
                          </span>
                        ))}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="card-footer">
              <div className="alert alert-info" style={{ margin: 0 }}>
                <span className="alert-icon">ℹ️</span>
                <span className="text-xs">{latestEval.evaluation_note}</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
