# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements/ /app/requirements/
RUN pip install --upgrade pip \
    && pip install -r requirements/dev.txt

COPY . /app

EXPOSE 8000

ENTRYPOINT ["python", "docker-entrypoint.py"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
