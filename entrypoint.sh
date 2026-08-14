#!/bin/sh
set -e

DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"

# ── Wait for PostgreSQL ───────────────────────────────────────────────────────
echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
while ! nc -z "${DB_HOST}" "${DB_PORT}"; do
  sleep 0.5
done
echo "PostgreSQL is ready."

# ── Apply database migrations ─────────────────────────────────────────────────
echo "Applying database migrations..."
python manage.py migrate --no-input

# ── Collect static files ──────────────────────────────────────────────────────
echo "Collecting static files..."
python manage.py collectstatic --no-input

# ── Start the application ─────────────────────────────────────────────────────
echo "Starting application..."

if [ "$1" = "celery" ]; then
  echo "Starting Celery worker..."
  exec "$@"
fi

# If DEVELOPMENT mode is enabled, use runserver with auto-reload
if [ "$DEVELOPMENT" = "True" ]; then
  echo "🚀 Development mode - Using Django runserver (no autoreload — evita OOM no Docker)"
  python manage.py runserver 0.0.0.0:8000 --noreload
else
  echo "🔥 Production mode - Using Gunicorn"
  exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 1 --timeout 120
fi
