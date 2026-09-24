from .base import *

DEBUG = True

if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# O Vite troca de porta sozinho (5173 -> 5174...) se a padrão estiver ocupada;
# em dev, libera localhost em qualquer porta pra não quebrar com CORS toda vez.
CORS_ALLOWED_ORIGIN_REGEXES = [r"^http://(localhost|127\.0\.0\.1):\d+$"]
