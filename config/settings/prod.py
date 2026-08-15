from .base import *

DEBUG = False

if not ALLOWED_HOSTS:
    raise ValueError(
        "DJANGO_ALLOWED_HOSTS precisa estar definido no .env em produção."
    )

SECURE_SSL_REDIRECT = env.bool("DJANGO_SECURE_SSL_REDIRECT", default=True)  # noqa: F405
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 60 * 60 * 24 * 7
SECURE_HSTS_INCLUDE_SUBDOMAINS = True

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    *MIDDLEWARE[1:],
]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
if not EMAIL_HOST_USER or not EMAIL_HOST_PASSWORD:  # noqa: F405
    raise ValueError(
        "EMAIL_HOST_USER e EMAIL_HOST_PASSWORD precisam estar definidos no "
        ".env em produção — é como o sistema envia a senha inicial das "
        "contas criadas pelo CSAdmin."
    )
