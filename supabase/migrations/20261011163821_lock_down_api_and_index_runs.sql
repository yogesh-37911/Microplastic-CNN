-- Keep application tables inaccessible to browser API roles. The Python backend
-- uses its private Postgres connection and Supabase Storage secret key.
CREATE POLICY analyses_backend_only ON public.analyses
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY training_runs_backend_only ON public.training_runs
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY evaluations_backend_only ON public.evaluations
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
CREATE POLICY dataset_manifests_backend_only ON public.dataset_manifests
    FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

CREATE INDEX IF NOT EXISTS evaluations_run_id_idx ON public.evaluations(run_id);
