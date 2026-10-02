"""Configuração do gunicorn para produção.

Respostas do chat são streams SSE longos: com workers `sync`, cada stream
ocuparia um worker inteiro. `gthread` atende várias conexões por worker, e o
timeout alto evita matar um stream lento do LLM no meio.

Uso: gunicorn config.wsgi:application -c gunicorn.conf.py
"""

import os

bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")
worker_class = "gthread"
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
