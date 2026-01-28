# Railway Multi-Service Template (Tidepool)

This repo is a **two-service** Railway template:

- **tidepool-query** (public, read-only)
- **tidepool-ingest** (private, write + compaction)

Railway templates for multiple services are created from a Railway project in
the UI rather than fully defined in `railway.json`/`railway.toml`. citeturn2search2turn3search0

## One-Click Template Setup

1. Create a Railway project from this GitHub repo.
2. Add two services:
   - `tidepool-query` → Config file `/query/railway.toml` (absolute path). citeturn2search1turn2search2
   - `tidepool-ingest` → Config file `/ingest/railway.toml` (absolute path). citeturn2search1turn2search2
3. For both services, set **Root Directory** to `/` (repo root). citeturn2search3
4. Add Railway Object Storage (S3-compatible bucket).
5. Attach the bucket env vars to both services:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_ENDPOINT_URL`
   - `AWS_REGION`
   - `BUCKET_NAME`
6. Optional: add a volume at `/data` for query caching.
7. Publish the template from the project’s Settings page.

## Troubleshooting

- If you see `stat /build/cmd/...: directory not found`, the service Root Directory
  is not set to `/` and the build context doesn’t include the repo root. citeturn2search3
