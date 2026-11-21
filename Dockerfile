# CPU inference image: API + demo UI. Expects ./bundle.pt (copy it from runs/<run>/bundle.pt).
#   cp runs/densenet121_128/bundle.pt bundle.pt
#   docker build -t cxr-triage . && docker run -p 8080:8080 cxr-triage
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,llm]"
COPY bundle.pt /app/bundle.pt
ENV CXR_BUNDLE=/app/bundle.pt PORT=8080 OMP_NUM_THREADS=2
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn cxr_triage.serve:app --host 0.0.0.0 --port ${PORT}"]
