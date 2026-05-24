@echo off
title VisionDx API - Dev Server

echo.
echo  =============================================
echo   VisionDx API - Starting Development Server
echo  =============================================
echo.

REM Move to the backend folder
cd /d "%~dp0"

REM Check if .env exists
if not exist ".env" (
    echo [ERROR] .env file not found. Copy .env.example to .env first.
    pause
    exit /b 1
)

REM Create tables if database doesn't exist
if not exist "visiondx_dev.db" (
    echo Creating database tables...
    python create_tables.py
)

echo Starting FastAPI server on http://localhost:8000
echo Swagger docs: http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
