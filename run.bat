@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"

call :load_env "%ROOT%.env"

if not defined DATABASE_URL set "DATABASE_URL=postgresql://postgres:1234@127.0.0.1:5432/partial_discharge_diagnosis"
if not defined DB_CONTAINER set "DB_CONTAINER=partial-discharge-rag-postgres-5432"
if not defined DB_IMAGE set "DB_IMAGE=pgvector/pgvector:pg16"
if not defined DB_POSTGRES_USER set "DB_POSTGRES_USER=postgres"
if not defined DB_POSTGRES_PASSWORD set "DB_POSTGRES_PASSWORD=1234"
if not defined DB_POSTGRES_DB set "DB_POSTGRES_DB=partial_discharge_diagnosis"
if not defined BACKEND_HOST set "BACKEND_HOST=127.0.0.1"
if not defined BACKEND_PORT set "BACKEND_PORT=8001"
if not defined FRONTEND_HOST set "FRONTEND_HOST=127.0.0.1"
if not defined FRONTEND_PORT set "FRONTEND_PORT=5173"
if not defined VITE_API_BASE set "VITE_API_BASE=http://%BACKEND_HOST%:%BACKEND_PORT%"

echo Starting Partial Discharge Diagnosis services...
echo.

where docker >nul 2>nul
if errorlevel 1 (
    echo [WARN] Docker was not found. PostgreSQL will not be started automatically.
    echo [WARN] Install Docker Desktop or start PostgreSQL manually on localhost:5432.
) else (
    powershell -NoProfile -Command "if ((Test-NetConnection 127.0.0.1 -Port 5432 -InformationLevel Quiet)) { exit 0 } else { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        echo PostgreSQL is already listening on 127.0.0.1:5432.
        goto start_services
    )

    docker container inspect "%DB_CONTAINER%" >nul 2>nul
    if errorlevel 1 (
        echo Creating PostgreSQL container: %DB_CONTAINER%
        docker run -d --name "%DB_CONTAINER%" ^
            -e POSTGRES_USER=%DB_POSTGRES_USER% ^
            -e POSTGRES_PASSWORD=%DB_POSTGRES_PASSWORD% ^
            -e POSTGRES_DB=%DB_POSTGRES_DB% ^
            -p 5432:5432 ^
            "%DB_IMAGE%"
    ) else (
        echo Starting PostgreSQL container: %DB_CONTAINER%
        docker start "%DB_CONTAINER%" >nul
    )

    echo Waiting for PostgreSQL to become ready...
    for /l %%i in (1,1,30) do (
        docker exec "%DB_CONTAINER%" pg_isready -U "%DB_POSTGRES_USER%" -d "%DB_POSTGRES_DB%" >nul 2>nul
        if not errorlevel 1 goto db_ready
        ping -n 2 127.0.0.1 >nul
    )

    echo [WARN] PostgreSQL did not report ready within 30 seconds.
    goto start_services

    :db_ready
    echo PostgreSQL is ready.
)

:start_services
echo Initializing RAG database schema...
cmd /c "cd /d ""%ROOT%"" && if exist "".venv\Scripts\activate.bat"" call "".venv\Scripts\activate.bat"" && python -m service.backend.scripts.rag_init_db"
if errorlevel 1 (
    echo [WARN] RAG database initialization failed. Backend may start in degraded mode.
)

echo Releasing backend/frontend ports if older dev servers are still running...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports = @([int]$env:BACKEND_PORT, [int]$env:FRONTEND_PORT); foreach ($port in $ports) { Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }" >nul 2>nul

start "PD Backend" cmd /k "cd /d ""%ROOT%"" && if exist "".venv\Scripts\activate.bat"" call "".venv\Scripts\activate.bat"" && python -m uvicorn service.backend.app.main:app --host %BACKEND_HOST% --port %BACKEND_PORT% --reload"

start "PD Frontend" cmd /k "cd /d ""%ROOT%service\frontend"" && set ""VITE_API_BASE=%VITE_API_BASE%"" && npm run dev -- --host %FRONTEND_HOST% --port %FRONTEND_PORT% --strictPort"

echo.
echo Backend:  http://%BACKEND_HOST%:%BACKEND_PORT%
echo Frontend: http://%FRONTEND_HOST%:%FRONTEND_PORT%
echo API base: %VITE_API_BASE%
echo Database: 127.0.0.1:5432 container=%DB_CONTAINER%
echo.
echo Started in separate terminal windows. Close those windows to stop backend/frontend.
echo Stop database with: docker stop %DB_CONTAINER%

endlocal
exit /b 0

:load_env
if not exist "%~1" exit /b 0
for /f "usebackq tokens=1,* delims==" %%A in ("%~1") do (
    set "ENV_KEY=%%~A"
    set "ENV_VALUE=%%~B"
    if not "!ENV_KEY!"=="" if not "!ENV_KEY:~0,1!"=="#" (
        set "!ENV_KEY!=!ENV_VALUE!"
    )
)
exit /b 0
