#!/usr/bin/env bash
# Deploy the demo to Google Cloud Run (CPU only, scales to zero, no GPU cost).
#
# Usage:  deploy/cloudrun.sh runs/densenet121_128/bundle.pt [region]
# Needs:  gcloud CLI, logged in (`gcloud auth login`), a project with billing enabled
#         (`gcloud config set project YOUR_PROJECT_ID`).
set -euo pipefail

BUNDLE="${1:?usage: deploy/cloudrun.sh path/to/bundle.pt [region]}"
REGION="${2:-us-central1}"
SERVICE="${SERVICE:-cxr-triage}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
[[ -n "$PROJECT" ]] || { echo "Run: gcloud config set project YOUR_PROJECT_ID" >&2; exit 1; }
[[ -f "$BUNDLE" ]] || { echo "Bundle not found: $BUNDLE (run cxr-evaluate first)" >&2; exit 1; }

cd "$(dirname "$0")/.."
cp "$BUNDLE" bundle.pt
trap 'rm -f bundle.pt' EXIT

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# --max-instances caps cost if the link gets popular; --min-instances 0 means $0 when idle
# (the first request after idling takes ~10-20 s while the container starts).
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 4 \
  --min-instances 0 --max-instances 3 \
  --timeout 60

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo
echo "Live at: $URL"
echo "Health:  curl $URL/health"
