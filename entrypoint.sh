#!/bin/sh
set -e

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

# If DEVELOPMENT mode is enabled, use runserver with auto-reload
if [ "$DEVELOPMENT" = "True" ]; then
  echo "🚀 Development mode - Using Django runserver with auto-reload"
  python manage.py runserver 0.0.0.0:8000
else
  echo "🔥 Production mode - Using Gunicorn"
  exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120
fi
