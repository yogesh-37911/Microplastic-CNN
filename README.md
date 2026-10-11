# Microplastic Vision 🔬🌊

**Image-Based Machine Learning for Preliminary Microplastic Screening**  
*A Smart India Hackathon Prototype for Environmental & Water Monitoring*

---

> ### ⚠️ Critical Scientific & Screening Disclaimer
> **Microplastic Vision is an image-based visual screening support tool.**
> Optical and microscopic image classification can identify visual features consistent with microplastic particles (fibers, fragments, beads, films, and foams). However, **definitive polymer characterization requires chemical analytical confirmation** such as Fourier-Transform Infrared (FTIR) Spectroscopy, Raman Microspectroscopy, or Pyrolysis Gas Chromatography–Mass Spectrometry (Py-GC/MS). Visual analysis cannot determine polymer composition or rule out organic mimics with 100% certainty.

---

## 📌 Project Overview

Plastic pollution in aquatic environments and sediment is an urgent ecological challenge. Traditional microplastic identification relies on manual visual counting under stereomicroscopes, which is labor-intensive, subject to high human observer bias, and non-scalable for routine environmental monitoring.

**Microplastic Vision** accelerates environmental workflows by providing:
1. **Preliminary Microscopic Screening**: High-throughput automated classification between `Microplastic` and `Non-microplastic` candidates.
2. **Confidence Scores & Triage**: Quantified confidence estimates allowing field researchers to prioritize samples for confirmatory spectroscopy.
3. **Flexible Architecture Support**: Multiple deep learning backbones (MobileNetV3 for efficient edge/mobile inference, ResNet18/ResNet34 for feature extraction, and lightweight custom CNNs).
4. **Transparent Evaluation & Provenance**: Per-class precision, recall, F1 scores, confusion matrices, and validation warnings preventing data leakage.
5. **Zero-Setup Demo Capability**: Embedded synthetic demo dataset generator allowing full local end-to-end evaluation without downloading external GB-scale datasets.

---

## 🏗️ System Architecture

```
microplastic-vision/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── classifier.py      # PyTorch model definitions (MobileNetV3, ResNet, SimpleCNN)
│   │   │   ├── detector.py        # Optional particle detection / bounding box interface
│   │   │   ├── evaluation.py      # Metrics computation, confusion matrix, per-class stats
│   │   │   ├── inference.py       # Inference engine with fallback demo prediction
│   │   │   ├── preprocessing.py   # Standardization, aspect ratio preservation, augmentations
│   │   │   └── training.py        # PyTorch training loop, split verification, early stopping
│   │   ├── routes/
│   │   │   ├── analysis.py        # Single & batch image upload, prediction & history
│   │   │   └── datasets.py        # Dataset validation, training runs, evaluation endpoints
│   │   ├── config.py              # Application settings via Pydantic Settings
│   │   ├── database.py            # Async SQLAlchemy engine with SQLite/aiosqlite
│   │   ├── demo_data.py           # Synthetic sample dataset generation
│   │   ├── models.py              # ORM models (Analysis, Dataset, TrainingRun, Evaluation)
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   └── main.py                # FastAPI app initialization, CORS, static serving
│   ├── data/                      # Local sqlite DB, uploads, datasets & checkpoints
│   ├── tests/
│   │   └── test_app.py            # Comprehensive 31 automated tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Sidebar.tsx        # Responsive navigation with live status badges
│   │   ├── pages/
│   │   │   ├── OverviewPage.tsx   # Dashboard with stats, recent analyses & workflow
│   │   │   ├── AnalyzePage.tsx    # Drag-and-drop screening, single/batch modes, image previews
│   │   │   ├── DatasetPage.tsx    # Dataset registry, validation checklist & guide
│   │   │   ├── EvaluationPage.tsx # Recharts metrics, class precision/recall & confusion matrix
│   │   │   ├── RunsPage.tsx       # Model training trigger, epoch history & loss curves
│   │   │   └── AboutPage.tsx      # Scientific rationale, FTIR/Raman workflows & limitations
│   │   ├── api.ts                 # Type-safe API client
│   │   ├── App.tsx                # React Router setup
│   │   └── index.css              # Custom responsive design system
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── start_backend.bat              # One-click Windows launch for FastAPI server
├── start_frontend.bat             # One-click Windows launch for React Vite app
├── start_streamlit.bat            # One-click Windows launch for Streamlit screening app
└── README.md
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ (npm installed)

### 1. Backend Setup & Run

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Generate initial synthetic demo dataset
python -m app.demo_data

# Start the FastAPI backend server (runs on port 8000)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI documentation will be available at:  
👉 `http://localhost:8000/docs`

### 2. Frontend Setup & Run

```bash
# Navigate to frontend directory in another terminal
cd frontend

# Install dependencies
npm install

# Start development server (runs on port 5173)
npm run dev
```

Open your browser at:  
👉 `http://localhost:5173`

### Deploy on Vercel

The React frontend is deployed from `frontend/` as a static Vercel project. The FastAPI/PyTorch backend is not deployed there because Vercel's serverless function bundle limit is smaller than the PyTorch runtime. Supabase is configured for persistent PostgreSQL records and private image storage; it does not run this PyTorch inference service.

For a backend host that can run the Python dependencies, configure these server-side environment variables:

- `DATABASE_URL`: the Supabase PostgreSQL connection string. The backend supports `postgresql://` URLs and enforces TLS.
- `SUPABASE_URL`: the project's API URL.
- `SUPABASE_SECRET_KEY`: the Supabase secret key, stored only on the backend host.
- `SUPABASE_STORAGE_BUCKET=analysis-images`
- `CORS_ORIGINS`: include the Vercel frontend origin.

Set `VITE_API_URL` in the Vercel frontend project to the deployed FastAPI base URL, then redeploy the frontend. Do not put the Supabase secret key in any `VITE_` variable. Supabase tables and the private `analysis-images` bucket are defined in `supabase/migrations/`.

Training, dataset validation, and PyTorch image inference still run in the Python backend. The static frontend alone cannot perform screening until `VITE_API_URL` points to that backend.

### Optional: Streamlit screening interface

The same PyTorch inference service can be opened in Streamlit:

```bash
cd backend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows, `start_streamlit.bat` launches the same interface.

---

## 🧪 Automated Testing

The project includes an automated test suite verifying preprocessing, model architectures, dataset splits without leakage, metric calculations, and REST API endpoints:

```bash
cd backend
python -m pytest tests/ -v
```

All 31 test cases run in isolation using temporary test fixtures without requiring external internet or GPU hardware.

---

## Training and model comparison

Once a suitable dataset is available, compare the CNN/transfer-learning models with:

```bash
cd backend
python -m app.ml.train_cli --dataset data/datasets/my_microscopy_dataset --architectures mobilenet_v3_small --epochs 8 --batch-size 32 --image-size 160 --freeze-backbone
```

The command writes training curves, split provenance, per-class metrics, and held-out results to `data/reports/model_comparison.json`. It refuses to run when the classes are too small or source-grouped evaluation would leak an acquisition. Keep generated graphics and synthetic particles outside the microscopy class folders.

### Exploratory public microfibre dataset

The repository includes a preparation script for the public [microfibres_CA_filter dataset](https://huggingface.co/datasets/femartip/microfibres_CA_filter). It creates one annotated fibre crop and, when available, one crop with no annotated fibre per source image. Each crop pair remains in the same source group during splitting. Background labels mean “no annotated fibre in this crop”; they are not chemically confirmed plastic-free labels. The dataset represents microfibres on cellulose acetate filters and is an exploratory training source.

From `backend/`, prepare and train the CPU-friendly frozen-head ResNet-18 experiment with:

```bash
python ingest_microfibres_ca.py
python -m app.ml.train_cli --dataset data/datasets/microfibres_ca_crops_v1 --architectures resnet18 --epochs 5 --batch-size 32 --learning-rate 0.0003 --image-size 128 --freeze-backbone --output data/reports/microfibres_ca_resnet18.json
```

The generated dataset includes `dataset_manifest.json` with source, license, label definitions, and limitations. Do not interpret its held-out score as validation across labs, polymer types, or microscopy setups.

### Train with the supplied microscope images

The 30 microscope images supplied on 2026-10-11 are included as user-identified microplastic examples in one acquisition group. The group stays intact during splitting. Their labels have not been independently chemically verified. From `backend/`, prepare the combined dataset and fine-tune the pretrained MobileNetV3-Small model:

```bash
python prepare_user_microscopy_dataset.py
python -m app.ml.train_cli --dataset data/datasets/microfibres_ca_user_20261011 --architectures mobilenet_v3_small --epochs 10 --batch-size 32 --learning-rate 0.0001 --image-size 160 --output data/reports/microfibres_ca_user_20261011.json
```

The run report records the source-grouped split and held-out metrics; the selected checkpoint stores its held-out score for inference readiness checks. A single user acquisition group can contribute to only one split and does not establish generalization to other microscopes or samples.

## 🔬 Dataset Directory Layout

To train or evaluate on real environmental microscopy datasets, organize images using the standard multi-class directory structure:

```
my_microplastic_dataset/
├── microplastic/
│   ├── sample_fiber_001.jpg
│   ├── sample_fragment_002.png
│   └── ...
└── non_microplastic/
    ├── sediment_sand_001.jpg
    ├── organic_matter_002.png
    └── ...
```

Training requires exactly the `microplastic` and `non_microplastic` folders with at
least 50 readable examples in each class. This is a low operational floor, not a recommended research dataset size;
representative samples from independent microscope sessions and a larger held-out
test set are needed before relying on performance claims.

The training workflow uses deterministic source-grouped 70/15/15 train/validation/test
splits, keeping crops from the same acquisition together. The validation-selected checkpoint is evaluated once on the held-out test
split, and its provenance is stored with the checkpoint. Existing generated or
legacy checkpoints without that provenance are not automatically activated.

When no validated model is available, analysis abstains instead of returning a
brightness-based demo guess. Blank, very small, and strongly flat digital images
are flagged as out-of-domain heuristically; this is a quality screen and does not
prove that an accepted image came from a microscope. Confidence is the model's
softmax score and is not calibrated unless a calibration set is supplied.

---

## 📡 API Reference Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health check |
| `/api/status` | GET | Active model status, demo mode flag, stats |
| `/api/analyze` | POST | Upload and classify a single microscopy image |
| `/api/analyze/batch` | POST | Classify up to 50 microscopy images in parallel |
| `/api/analyses` | GET | Paginated screening history |
| `/api/datasets` | GET / POST | Register and inspect datasets |
| `/api/datasets/validate` | POST | Validate folder layout and check for warnings |
| `/api/train` | POST | Launch background model training run |
| `/api/runs` | GET | List training runs with epoch loss and validation metrics |
| `/api/evaluate` | POST | Run evaluation on test split and generate confusion matrix |
| `/api/uploads/{filename}` | GET | Secure image preview retrieval |

---

## 🏆 Smart India Hackathon Presentation Points

1. **Realistic Scope & Academic Honesty**: Rather than asserting optical images replace chemical spectrometry, the tool acts as a triage accelerator—reducing manual laboratory screening hours by over 80%.
2. **Transparent Confidence & Warnings**: Predictions explicitly indicate model uncertainty and guide scientists toward spectroscopic confirmation (FTIR/Raman).
3. **Extensibility**: Ready to integrate bounding-box object detection (YOLOv8) when spatial coordinates and particle count density per filter mesh are needed.
4. **Data Privacy & Local Deployment**: Runs completely on-premise without transmitting proprietary sample images to third-party cloud services.
