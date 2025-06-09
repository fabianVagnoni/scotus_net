# SCOTUS AI Project Setup Script for Windows
# Run this script in PowerShell to set up the project environment

Write-Host "Setting up SCOTUS AI Project Environment" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install spaCy model
Write-Host "Installing spaCy English model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm

# Create directories
Write-Host "Creating project directories..." -ForegroundColor Yellow
$directories = @(
    "data\raw",
    "data\processed", 
    "data\external",
    "logs",
    "models_output",
    "notebooks",
    ".cache"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "   Created: $dir" -ForegroundColor Cyan
}

# Copy environment example
if (Test-Path "env.example") {
    if (-not (Test-Path ".env")) {
        Copy-Item "env.example" ".env"
        Write-Host "Created .env file from env.example" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Configure your .env file with API keys and settings"
Write-Host "2. Run data scraper: python src/data_pipeline/scraper.py"
Write-Host "3. Process data: python src/data_pipeline/processor.py"
Write-Host "4. Train model: python src/models/model_trainer.py"
Write-Host ""
Write-Host "To activate the environment later, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan 