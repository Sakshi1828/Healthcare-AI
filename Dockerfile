# CPU inference image. Build after `cxr-evaluate` has produced runs/<run>/bundle.pt:
#   docker build --build-arg BUNDLE=runs/densenet121_128/bundle.pt -t cxr-triage .
#   docker run -p 8080:8080 cxr-triage
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
ARG BUNDLE=bundle.pt
COPY ${BUNDLE} /app/bundle.pt
ENV CXR_BUNDLE=/app/bundle.pt PORT=8080
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn cxr_triage.serve:app --host 0.0.0.0 --port ${PORT}"]
