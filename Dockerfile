FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000 \
    BEE_DATA_DIR=/data

WORKDIR /app

# CPU-only PyTorch (much smaller than the default CUDA build), then the rest
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# whole repo: app.py, bee_pages.html, favicon.ico, tokenizer + checkpoints (next to app.py or in models/)
COPY . .

# gunicorn entry point, created here so the build never depends on a separate wsgi.py in the repo
RUN printf '%s\n' \
  'import argparse' \
  'import app as bee' \
  'bee.init_engine(argparse.Namespace(tokenizer=None, chat=None, tool=None, cpu=True))' \
  'bee.init_db()' \
  'application = bee.app' > wsgi.py

# non-root user; /data holds logs (bee_data.json) and the session secret
RUN useradd -m bee && mkdir -p /data && chown -R bee:bee /data /app
USER bee
VOLUME /data

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','5000')+'/api/info')"

# ONE worker: models and logs live in this process. Threads let several visitors connect.
CMD ["sh", "-c", "exec gunicorn wsgi:application -b 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 300"]
