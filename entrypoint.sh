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
exec "$@"
