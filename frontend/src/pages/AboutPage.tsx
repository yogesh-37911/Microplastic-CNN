export default function AboutPage() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">About & Limitations</h1>
        <p className="page-subtitle">
          Understanding what Microplastic Vision can and cannot do.
        </p>
      </div>

      {/* Intended use */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Intended Use</h2>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8 }}>
          <p>
            Microplastic Vision is a <strong>preliminary image-based screening tool</strong> that
            uses computer vision and deep learning to classify microscopy or laboratory images as
            potential microplastic or non-microplastic particles based on visual appearance and morphology.
          </p>
          <p className="mt-md">
            It is designed for use in <strong>research and environmental monitoring settings</strong>
            where rapid visual screening of large numbers of images is needed before detailed chemical
            analysis.
          </p>
        </div>
      </div>

      {/* Scientific Limitations */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">⚠️ Scientific Limitations</h2>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8 }}>
          <div className="alert alert-warning mb-lg">
            <span className="alert-icon">⚠️</span>
            <div>
              <strong>Critical Distinction:</strong> Image-based classification identifies visual features
              and morphology. It does <strong>NOT</strong> identify the chemical composition or polymer type
              of a particle.
            </div>
          </div>

          <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>What this tool does NOT do:</h3>
          <ul style={{ paddingLeft: '20px', marginBottom: '16px' }}>
            <li>Confirm that a particle is definitively plastic</li>
            <li>Identify polymer type (PET, PE, PP, etc.) from ordinary images</li>
            <li>Replace FTIR, Raman spectroscopy, or other chemical analysis</li>
            <li>Provide laboratory-validated results</li>
            <li>Account for all environmental matrices and preparation methods</li>
          </ul>

          <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Results depend on:</h3>
          <ul style={{ paddingLeft: '20px', marginBottom: '16px' }}>
            <li>Image quality and resolution</li>
            <li>Microscope type, magnification, and settings</li>
            <li>Lighting conditions and background</li>
            <li>Sample preparation method</li>
            <li>Particle size, color, and transparency</li>
            <li>Quality and diversity of training data</li>
            <li>Class balance in the training dataset</li>
          </ul>

          <h3 style={{ fontWeight: 700, marginBottom: '8px' }}>Confidence scores:</h3>
          <p>
            The confidence score reflects the model's certainty about its visual classification.
            It is <strong>not</strong> the probability that a particle is chemically a polymer.
            Low-confidence results should be reviewed manually; high-confidence results still
            require chemical confirmation for definitive identification.
          </p>
        </div>
      </div>

      {/* Data Requirements */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Data Requirements</h2>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8 }}>
          <p>For meaningful model training, you need:</p>
          <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
            <li>Labelled microscopy images organized into class directories</li>
            <li>Sufficient images per class (recommended: hundreds to thousands)</li>
            <li>Representative samples from your target environment and microscope</li>
            <li>Consistent imaging conditions or documented variation</li>
            <li>Verified labels (ideally confirmed by chemical analysis for a subset)</li>
          </ul>
          <div className="alert alert-info mt-md">
            <span className="alert-icon">ℹ️</span>
            <span className="text-sm">
              The demo mode uses synthetic images for functional testing.
              These do NOT represent real microplastics. Any model trained on demo data
              must not be used for scientific analysis.
            </span>
          </div>
        </div>
      </div>

      {/* YOLO Detection */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Optional: YOLO Particle Detection</h2>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8 }}>
          <p>
            When a trained YOLO detector and bounding-box annotation data are available,
            Microplastic Vision can locate individual candidate particles in images.
          </p>
          <p className="mt-md">
            This feature requires:
          </p>
          <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
            <li>Ultralytics YOLO package installed</li>
            <li>A trained detector weight file (.pt)</li>
            <li>Bounding-box annotated training data</li>
          </ul>
          <p className="mt-md text-sm text-secondary">
            YOLO detection is optional and does not affect the core classification workflow.
            The application runs fully without it.
          </p>
        </div>
      </div>

      {/* Chemical Confirmation */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Path to Chemical Confirmation</h2>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            {[
              {
                step: '1',
                title: 'Visual Screening',
                desc: 'Image classification for preliminary sorting (this tool)',
                status: '✅ Available',
              },
              {
                step: '2',
                title: 'Manual Review',
                desc: 'Expert review of flagged particles',
                status: '👤 Manual step',
              },
              {
                step: '3',
                title: 'Chemical Analysis',
                desc: 'FTIR, Raman, or pyrolysis-GC/MS for polymer identification',
                status: '🔬 Laboratory',
              },
              {
                step: '4',
                title: 'Confirmed Result',
                desc: 'Definitive polymer type and characterisation',
                status: '✓ Ground truth',
              },
            ].map((s) => (
              <div key={s.step} style={{
                padding: '20px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--color-neutral-50)',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '14px', fontWeight: 700, marginBottom: '8px' }}>
                  Step {s.step}: {s.title}
                </div>
                <div className="text-xs text-secondary" style={{ marginBottom: '8px' }}>{s.desc}</div>
                <div className="text-xs" style={{ color: 'var(--color-primary-600)' }}>{s.status}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="card mb-lg">
        <div className="card-header">
          <h2 className="card-title">Technical Details</h2>
        </div>
        <div className="card-body">
          <table className="data-table">
            <tbody>
              <tr><td className="font-bold">Frontend</td><td>React, TypeScript, Vite, Recharts</td></tr>
              <tr><td className="font-bold">Backend</td><td>Python, FastAPI, SQLAlchemy, SQLite</td></tr>
              <tr><td className="font-bold">ML Framework</td><td>PyTorch, torchvision, scikit-learn</td></tr>
              <tr><td className="font-bold">Available Models</td><td>SimpleCNN, ResNet-18, MobileNet-V2</td></tr>
              <tr><td className="font-bold">Optional Detection</td><td>Ultralytics YOLO (when configured)</td></tr>
              <tr><td className="font-bold">Image Formats</td><td>JPG, JPEG, PNG, TIFF, BMP</td></tr>
              <tr><td className="font-bold">Default Input Size</td><td>224×224 pixels</td></tr>
              <tr><td className="font-bold">API Documentation</td><td>
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
                  OpenAPI Docs (localhost:8000/docs)
                </a>
              </td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Version */}
      <div className="card">
        <div className="card-body" style={{ textAlign: 'center' }}>
          <p className="text-sm text-muted">
            Microplastic Vision v0.1.0 · Smart India Hackathon 2026 · Prototype
          </p>
          <p className="text-xs text-muted mt-md">
            This is a research prototype. Results should not be used as sole evidence
            for environmental or regulatory decisions without independent verification.
          </p>
        </div>
      </div>
    </div>
  );
}
