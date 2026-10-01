# syntax=docker/dockerfile:1
# TrueEdit OCR — production image
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
# Optional build secret "extra_ca": extra CA certificate for builds behind a TLS-inspecting proxy.
RUN --mount=type=secret,id=extra_ca,required=false \
    apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/* \
 && if [ -f /run/secrets/extra_ca ]; then cat /run/secrets/extra_ca >> /etc/ssl/certs/ca-certificates.crt; fi
COPY requirements.txt requirements-models.txt ./
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --cert /etc/ssl/certs/ca-certificates.crt -r requirements.txt \
 && /opt/venv/bin/pip install --no-cache-dir --cert /etc/ssl/certs/ca-certificates.crt --no-deps -r requirements-models.txt
ENV PATH=/opt/venv/bin:$PATH
COPY trueedit ./trueedit
COPY web ./web
COPY samples ./samples
ENV HOST=0.0.0.0 PORT=8000 TRUEEDIT_DATA=/data/projects TRUEEDIT_RETENTION_DAYS=2 PYTHONUNBUFFERED=1
# fail the build if the OCR models cannot load
RUN python -c "from trueedit.ocr.ppocr import load_models; m = load_models(); assert m['rec4'] is not None"
EXPOSE 8000
CMD ["python", "-m", "trueedit.app"]
