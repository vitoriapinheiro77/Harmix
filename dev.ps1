# Development server startup script for Harmix
# Usage: .\dev.ps1

# Colors for output
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Cyan = [System.ConsoleColor]::Cyan

Write-Host "=== Harmix Development Server ===" -ForegroundColor $Cyan

# Check if venv exists
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor $Yellow
    python -m venv venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor $Cyan
& .\venv\Scripts\Activate.ps1

# Run migrations
Write-Host "`nRunning migrations..." -ForegroundColor $Cyan
python manage.py migrate --noinput

# Collect static files
Write-Host "Collecting static files..." -ForegroundColor $Cyan
python manage.py collectstatic --noinput

# Start development server
Write-Host "`n=== Server Starting on http://127.0.0.1:8000 ===" -ForegroundColor $Green
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor $Yellow

python manage.py runserver 0.0.0.0:8000
