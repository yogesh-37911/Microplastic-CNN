import { useEffect, useState } from 'react';
import { getDatasets, validateDataset, type DatasetInfo } from '../api';

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [validatePath, setValidatePath] = useState('');
  const [validateName, setValidateName] = useState('');
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = () => {
    setLoading(true);
    getDatasets()
      .then(setDatasets)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const handleValidate = async () => {
    if (!validatePath.trim()) return;
    setValidating(true);
    setError('');
    try {
      await validateDataset(validatePath.trim(), validateName.trim() || undefined);
      setValidatePath('');
      setValidateName('');
      fetchDatasets();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setValidating(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <span>Loading datasets…</span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Dataset Management</h1>
        <p className="page-subtitle">
          Register and validate datasets for training and evaluation.
        </p>
      </div>

      {/* Directory layout documentation */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Expected Dataset Layout</h2>
        </div>
        <div className="card-body">
          <pre style={{
            background: 'var(--color-neutral-50)',
            padding: 'var(--spacing-md)',
            borderRadius: 'var(--radius-md)',
            fontSize: 'var(--font-size-sm)',
            lineHeight: 1.8,
            overflow: 'auto',
          }}>
{`data/datasets/my_dataset/
  ├── microplastic/
  │   ├── image_001.png
  │   ├── image_002.jpg
  │   └── ...
  └── non_microplastic/
      ├── image_001.png
      ├── image_002.jpg
      └── ...`}
          </pre>
          <p className="text-sm text-secondary mt-md">
            Place images into class-named subdirectories. Supported formats: JPG, JPEG, PNG, TIFF, BMP.
          </p>
        </div>
      </div>

      {/* Validate form */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Validate Dataset</h2>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label htmlFor="dataset-path" className="form-label">Dataset Directory Path</label>
            <input
              id="dataset-path"
              type="text"
              className="form-input"
              placeholder="e.g. data/datasets/my_dataset"
              value={validatePath}
              onChange={(e) => setValidatePath(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor="dataset-name" className="form-label">Dataset Name (optional)</label>
            <input
              id="dataset-name"
              type="text"
              className="form-input"
              placeholder="e.g. Lab samples batch 1"
              value={validateName}
              onChange={(e) => setValidateName(e.target.value)}
            />
          </div>
          <button
            className="btn btn-primary"
            onClick={handleValidate}
            disabled={validating || !validatePath.trim()}
          >
            {validating ? 'Validating…' : 'Validate & Register'}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-lg">
          <span className="alert-icon">❌</span>
          {error}
        </div>
      )}

      {/* Datasets list */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Registered Datasets</h2>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          {datasets.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📁</div>
              <div className="empty-state-text">No datasets registered</div>
              <div className="empty-state-hint">
                Validate a dataset directory above to register it.
              </div>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Total Images</th>
                  <th>Classes</th>
                  <th>Splits</th>
                  <th>Status</th>
                  <th>Warnings</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.id}>
                    <td>
                      <div className="font-bold">{d.name}</div>
                      <div className="text-xs text-muted">{d.base_path}</div>
                    </td>
                    <td>{d.total_images}</td>
                    <td>
                      {d.class_counts && Object.entries(d.class_counts).map(([cls, cnt]) => (
                        <div key={cls} className="text-sm">
                          <span className={`badge ${cls === 'microplastic' ? 'badge-error' : 'badge-success'}`} style={{ marginRight: 4 }}>
                            {cls.replace('_', ' ')}
                          </span>
                          <span className="text-muted">{cnt}</span>
                        </div>
                      ))}
                    </td>
                    <td>
                      {d.training_ready && d.split_counts && Object.entries(d.split_counts).map(([split, cnt]) => (
                        <div key={split} className="text-xs">
                          {split}: {cnt}
                        </div>
                      ))}
                    </td>
                    <td>
                      <span className={`badge ${d.training_ready ? 'badge-success' : 'badge-warning'}`}>
                        {d.training_ready ? 'Ready for training' : 'Not training ready'}
                      </span>
                    </td>
                    <td>
                      {d.validation_warnings && d.validation_warnings.length > 0 ? (
                        <ul className="warnings-list">
                          {d.validation_warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      ) : (
                        <span className="text-muted text-xs">None</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
