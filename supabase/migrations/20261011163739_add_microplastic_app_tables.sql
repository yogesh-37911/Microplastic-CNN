-- Persistent application data for Microplastic Vision.
CREATE TABLE public.analyses (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    original_filename text NOT NULL,
    storage_path text NOT NULL,
    file_type text NOT NULL,
    image_width integer,
    image_height integer,
    predicted_class text,
    confidence double precision,
    model_name text,
    model_version text,
    status text NOT NULL DEFAULT 'pending',
    error_message text,
    is_demo boolean NOT NULL DEFAULT false,
    warnings jsonb,
    preprocessing_config jsonb,
    detections jsonb,
    detection_count integer
);

CREATE TABLE public.training_runs (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    run_name text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    model_architecture text NOT NULL,
    dataset_manifest text,
    dataset_source text,
    epochs integer NOT NULL DEFAULT 25,
    batch_size integer NOT NULL DEFAULT 16,
    learning_rate double precision NOT NULL DEFAULT 0.001,
    image_size integer NOT NULL DEFAULT 224,
    class_weights jsonb,
    augmentation_config jsonb,
    preprocessing_config jsonb,
    seed integer DEFAULT 42,
    checkpoint_path text,
    best_val_accuracy double precision,
    training_history jsonb,
    class_mapping jsonb,
    train_count integer,
    val_count integer,
    test_count integer,
    split_method text,
    test_metrics jsonb,
    error_message text,
    progress double precision DEFAULT 0.0
);

CREATE TABLE public.evaluations (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    run_id text NOT NULL REFERENCES public.training_runs(id) ON DELETE CASCADE,
    test_split_name text NOT NULL DEFAULT 'test',
    test_count integer,
    class_counts jsonb,
    accuracy double precision,
    precision_per_class jsonb,
    recall_per_class jsonb,
    f1_per_class jsonb,
    precision_macro double precision,
    recall_macro double precision,
    f1_macro double precision,
    confusion_matrix jsonb,
    class_order jsonb,
    model_checkpoint text,
    preprocessing_config jsonb,
    dataset_manifest text,
    status text DEFAULT 'pending',
    error_message text
);

CREATE TABLE public.dataset_manifests (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    name text NOT NULL,
    description text,
    source text,
    license_notes text,
    base_path text NOT NULL,
    total_images integer DEFAULT 0,
    class_counts jsonb,
    split_counts jsonb,
    validation_warnings jsonb,
    label_provenance text,
    is_valid boolean DEFAULT false
);

ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dataset_manifests ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.analyses, public.training_runs, public.evaluations, public.dataset_manifests FROM anon, authenticated;

-- Images remain private; the backend uses its secret key for object access.
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('analysis-images', 'analysis-images', false, 52428800,
        ARRAY['image/jpeg', 'image/png', 'image/tiff', 'image/bmp']::text[])
ON CONFLICT (id) DO NOTHING;
