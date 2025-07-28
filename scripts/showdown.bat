@echo off
setlocal enabledelayedexpansion

:: Pokemon Showdown Server Manager for Windows
:: Usage: showdown.bat [start|stop|restart|status|quick] [num_servers]

set "PROJECT_ROOT=%~dp0.."
set "SHOWDOWN_DIR=%PROJECT_ROOT%\pokemon-showdown"
set "PID_DIR=%PROJECT_ROOT%\logs\pids"
set "LOG_DIR=%PROJECT_ROOT%\logs\showdown_logs"
set "TRAIN_CONFIG=%PROJECT_ROOT%\config\train_config.yml"
set "DEFAULT_SERVERS=5"
set "DEFAULT_PORT=8000"
set "MAX_CONNECTIONS=25"

:: Create directories if they don't exist
if not exist "%PID_DIR%" mkdir "%PID_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Get command
set "COMMAND=%1"
if "%COMMAND%"=="" (
    echo Usage: %~nx0 [start^|stop^|restart^|status^|quick] [num_servers]
    echo.
    echo Commands:
    echo   start [n]    Start n servers ^(default: %DEFAULT_SERVERS%^)
    echo   stop         Stop all servers
    echo   restart [n]  Restart with n servers
    echo   status       Show server status
    echo   quick        Auto-start based on train_config.yml
    echo   clean        Force stop all servers on ports 8000-8009
    exit /b 1
)

:: Function to find Python executable
call :find_python
if "%PYTHON_EXE%"=="" (
    echo Error: Python not found. Please install Python 3.x
    exit /b 1
)

:: Execute command
if /i "%COMMAND%"=="start" (
    set "NUM_SERVERS=%2"
    if "!NUM_SERVERS!"=="" set "NUM_SERVERS=%DEFAULT_SERVERS%"
    call :start_servers !NUM_SERVERS!
) else if /i "%COMMAND%"=="stop" (
    call :stop_servers
) else if /i "%COMMAND%"=="restart" (
    set "NUM_SERVERS=%2"
    if "!NUM_SERVERS!"=="" set "NUM_SERVERS=%DEFAULT_SERVERS%"
    call :stop_servers
    timeout /t 2 /nobreak >nul
    call :start_servers !NUM_SERVERS!
) else if /i "%COMMAND%"=="status" (
    call :show_status
) else if /i "%COMMAND%"=="quick" (
    call :quick_start
) else if /i "%COMMAND%"=="clean" (
    call :force_clean
) else (
    echo Invalid command: %COMMAND%
    exit /b 1
)

exit /b 0

:: ==================== Functions ====================

:find_python
:: Try to find Python in common locations
set "PYTHON_EXE="
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_EXE=%%p"
        goto :eof
    )
)
goto :eof

:start_servers
set "NUM=%1"
echo Starting %NUM% Pokemon Showdown servers...
echo.

:: Check if Node.js is installed
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: Node.js not found. Please install Node.js
    exit /b 1
)

:: Check if pokemon-showdown exists
if not exist "%SHOWDOWN_DIR%\pokemon-showdown" (
    echo Error: Pokemon Showdown not found at %SHOWDOWN_DIR%
    echo Please run: cd pokemon-showdown ^&^& npm install
    exit /b 1
)

set /a "STARTED=0"
for /l %%i in (0,1,%NUM%-1) do (
    set /a "PORT=%DEFAULT_PORT%+%%i"
    set "PID_FILE=%PID_DIR%\showdown_!PORT!.pid"
    
    :: Check if already running
    if exist "!PID_FILE!" (
        set /p PID=<"!PID_FILE!"
        tasklist /fi "PID eq !PID!" 2>nul | find "node.exe" >nul
        if !errorlevel!==0 (
            echo Port !PORT!: Already running ^(PID: !PID!^)
        ) else (
            del "!PID_FILE!"
            call :start_single_server !PORT!
            set /a "STARTED+=1"
        )
    ) else (
        call :start_single_server !PORT!
        set /a "STARTED+=1"
    )
)

echo.
echo Started %STARTED% servers
goto :eof

:start_single_server
set "PORT=%1"
set "LOG_FILE=%LOG_DIR%\showdown_server_%PORT%.log"
set "PID_FILE=%PID_DIR%\showdown_%PORT%.pid"

echo Starting server on port %PORT%...

:: Start server in background
cd /d "%SHOWDOWN_DIR%"
start /b cmd /c "node pokemon-showdown start --no-security --port %PORT% > "%LOG_FILE%" 2>&1"

:: Wait for server to start
timeout /t 2 /nobreak >nul

:: Get PID of the started process
for /f "tokens=2" %%a in ('tasklist /v /fo csv ^| findstr /i "pokemon-showdown.*%PORT%"') do (
    set "PID=%%~a"
    echo !PID!> "!PID_FILE!"
    echo Port %PORT%: Started ^(PID: !PID!^)
    goto :eof
)

:: If we couldn't find the process, try alternative method
for /f "skip=3 tokens=2" %%a in ('tasklist /fi "IMAGENAME eq node.exe" /fo csv') do (
    set "LAST_PID=%%~a"
)
if defined LAST_PID (
    echo !LAST_PID!> "!PID_FILE!"
    echo Port %PORT%: Started ^(PID: !LAST_PID!^)
)
goto :eof

:stop_servers
echo Stopping all Pokemon Showdown servers...
echo.

set /a "STOPPED=0"
:: First, try to stop managed servers (with PID files)
for /f %%f in ('dir /b "%PID_DIR%\showdown_*.pid" 2^>nul') do (
    set "PID_FILE=%PID_DIR%\%%f"
    set /p PID=<"!PID_FILE!"
    
    :: Extract port from filename
    for /f "tokens=2 delims=_." %%p in ("%%f") do set "PORT=%%p"
    
    :: Try to kill the process
    tasklist /fi "PID eq !PID!" 2>nul | find "node.exe" >nul
    if !errorlevel!==0 (
        taskkill /PID !PID! /F >nul 2>&1
        if !errorlevel!==0 (
            echo Port !PORT!: Stopped ^(PID: !PID!^)
            set /a "STOPPED+=1"
        ) else (
            echo Port !PORT!: Failed to stop ^(PID: !PID!^)
        )
    ) else (
        echo Port !PORT!: Not running ^(stale PID: !PID!^)
    )
    del "!PID_FILE!"
)

:: Kill any node processes listening on our ports (8000-8009)
echo.
echo Checking for unmanaged servers on ports 8000-8009...
for /l %%i in (0,1,9) do (
    set /a "PORT=8000+%%i"
    
    :: Find process using the port
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":!PORT!" ^| findstr "LISTENING"') do (
        set "PID=%%p"
        if "!PID!" NEQ "0" (
            :: Verify it's a node.exe process
            tasklist /fi "PID eq !PID!" 2>nul | find "node.exe" >nul
            if !errorlevel!==0 (
                echo Port !PORT!: Found unmanaged server ^(PID: !PID!^), stopping...
                taskkill /PID !PID! /F >nul 2>&1
                if !errorlevel!==0 (
                    echo Port !PORT!: Stopped unmanaged server
                    set /a "STOPPED+=1"
                ) else (
                    echo Port !PORT!: Failed to stop unmanaged server
                )
            )
        )
    )
)

:: Also kill any orphaned node processes running pokemon-showdown
for /f "tokens=2" %%p in ('tasklist /fi "IMAGENAME eq node.exe" /fo csv 2^>nul ^| findstr /i "pokemon-showdown"') do (
    taskkill /PID %%~p /F >nul 2>&1
    set /a "STOPPED+=1"
)

echo.
echo Stopped %STOPPED% servers
goto :eof

:show_status
echo Pokemon Showdown Server Status
echo ==============================
echo.

:: Check each port
set /a "RUNNING=0"
set /a "TOTAL=0"
for /l %%i in (0,1,9) do (
    set /a "PORT=%DEFAULT_PORT%+%%i"
    set "PID_FILE=%PID_DIR%\showdown_!PORT!.pid"
    set /a "TOTAL+=1"
    
    if exist "!PID_FILE!" (
        set /p PID=<"!PID_FILE!"
        tasklist /fi "PID eq !PID!" 2>nul | find "node.exe" >nul
        if !errorlevel!==0 (
            echo Port !PORT!: RUNNING ^(PID: !PID!^)
            set /a "RUNNING+=1"
            
            :: Show log file info
            if exist "%LOG_DIR%\showdown_server_!PORT!.log" (
                for %%s in ("%LOG_DIR%\showdown_server_!PORT!.log") do (
                    echo   Log: %%~zs bytes - %%~ts
                )
            )
        ) else (
            echo Port !PORT!: NOT RUNNING ^(stale PID file^)
            del "!PID_FILE!"
        )
    ) else (
        :: Check if port is in use
        netstat -an | findstr ":!PORT!" | findstr "LISTENING" >nul
        if !errorlevel!==0 (
            echo Port !PORT!: IN USE ^(unmanaged^)
        ) else (
            echo Port !PORT!: NOT RUNNING
        )
    )
)

echo.
echo Summary:
echo   Running servers: %RUNNING%
echo   Checked ports: %DEFAULT_PORT%-%PORT%
goto :eof

:quick_start
echo Auto-configuring servers based on train_config.yml...

:: Use Python to parse YAML and determine server count
set "TEMP_SCRIPT=%TEMP%\parse_servers.py"
(
echo import yaml
echo import sys
echo try:
echo     with open^(r'%TRAIN_CONFIG%', 'r'^) as f:
echo         config = yaml.safe_load^(f^)
echo     parallel = config.get^('parallel', 10^)
echo     servers_needed = max^(1, ^(parallel + 24^) // 25^)
echo     print^(servers_needed^)
echo except:
echo     print^(5^)
) > "%TEMP_SCRIPT%"

for /f %%n in ('"%PYTHON_EXE%" "%TEMP_SCRIPT%"') do set "NUM_SERVERS=%%n"
del "%TEMP_SCRIPT%"

echo Detected parallel=%NUM_SERVERS% environments, starting %NUM_SERVERS% servers...
call :start_servers %NUM_SERVERS%
goto :eof

:force_clean
echo Force cleaning all servers on ports 8000-8009...
echo.

set /a "KILLED=0"
:: Kill all node processes on our ports
for /l %%i in (0,1,9) do (
    set /a "PORT=8000+%%i"
    
    :: Find all processes using the port
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":!PORT!" ^| findstr "LISTENING"') do (
        set "PID=%%p"
        if "!PID!" NEQ "0" (
            echo Port !PORT!: Killing process ^(PID: !PID!^)
            taskkill /PID !PID! /F >nul 2>&1
            if !errorlevel!==0 (
                set /a "KILLED+=1"
            )
        )
    )
)

:: Clean up any PID files
if exist "%PID_DIR%\showdown_*.pid" (
    del /q "%PID_DIR%\showdown_*.pid" >nul 2>&1
    echo Cleaned up PID files
)

echo.
echo Force killed %KILLED% processes
goto :eof