@echo off
REM Pokemon Showdown Server Management Script for Windows
REM Usage: showdown.bat [start|stop|status|quick] [number_of_servers]

setlocal enabledelayedexpansion

REM Configuration
set "PROJECT_ROOT=%~dp0.."
set "SHOWDOWN_DIR=%PROJECT_ROOT%\pokemon-showdown"
set "LOGS_DIR=%PROJECT_ROOT%\logs\showdown_logs"
set "PID_DIR=%PROJECT_ROOT%\logs\pids"
set "CONFIG_FILE=%PROJECT_ROOT%\config\train_config.yml"

REM Create directories if they don't exist
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Parse command line arguments
set "COMMAND=%1"
set "NUM_SERVERS=%2"

if "%COMMAND%"=="" (
    call :show_help
    exit /b 1
)

if "%COMMAND%"=="start" goto :start_servers
if "%COMMAND%"=="stop" goto :stop_servers
if "%COMMAND%"=="status" goto :show_status
if "%COMMAND%"=="quick" goto :quick_start
if "%COMMAND%"=="help" goto :show_help

echo [ERROR] Unknown command: %COMMAND%
call :show_help
exit /b 1

:start_servers
if "%NUM_SERVERS%"=="" set "NUM_SERVERS=5"
echo [INFO] Starting %NUM_SERVERS% Pokemon Showdown servers...
echo [INFO] Project root: %PROJECT_ROOT%
echo [INFO] Port range: 8000-800%NUM_SERVERS%

REM Check if Node.js is available
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js first.
    exit /b 1
)

REM Check if pokemon-showdown directory exists
if not exist "%SHOWDOWN_DIR%" (
    echo [ERROR] Pokemon Showdown directory not found: %SHOWDOWN_DIR%
    echo [ERROR] Please make sure Pokemon Showdown is properly installed.
    exit /b 1
)

echo.
echo Starting servers...
echo ===============================================

for /l %%i in (0,1,%NUM_SERVERS%) do (
    set /a "port=8000+%%i"
    set /a "server_num=%%i+1"
    
    if !server_num! leq %NUM_SERVERS% (
        echo [INFO] Starting Pokemon Showdown server #!server_num! on port !port!...
        
        REM Start server in background and save PID
        start "" /b cmd /c "cd /d "%SHOWDOWN_DIR%" && node pokemon-showdown start --no-security --port !port! > "%LOGS_DIR%\showdown_server_!port!.log" 2>&1"
        
        REM Wait a moment for server to start
        timeout /t 1 /nobreak >nul
        
        REM Find the PID using netstat (approximate method)
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":!port! " ^| findstr "LISTENING"') do (
            echo !port!:%%a > "%PID_DIR%\showdown_!port!.pid"
            echo [OK] Server #!server_num! started successfully ^(Port: !port!, PID: %%a^)
            echo       Log file: %LOGS_DIR%\showdown_server_!port!.log
        )
    )
)

echo ===============================================
echo [SUCCESS] Successfully started %NUM_SERVERS% Pokemon Showdown server(s)
echo.
echo Server Status:
for /l %%i in (0,1,%NUM_SERVERS%) do (
    set /a "port=8000+%%i"
    set /a "server_num=%%i+1"
    
    if !server_num! leq %NUM_SERVERS% (
        netstat -an | findstr ":!port! " | findstr "LISTENING" >nul
        if !errorlevel! equ 0 (
            echo   [OK] Server #!server_num!: http://localhost:!port! - Running
        ) else (
            echo   [ERROR] Server #!server_num!: http://localhost:!port! - Failed to start
        )
    )
)

echo.
echo Management commands:
echo   * Check status: showdown.bat status
echo   * Stop all servers: showdown.bat stop
echo   * View logs: type "%LOGS_DIR%\showdown_server_[PORT].log"
echo.
echo [INFO] All servers are running in the background.
goto :eof

:stop_servers
echo [INFO] Stopping Pokemon Showdown servers...
echo Checking port range: 8000-8010
echo ===============================================

set "stopped_count=0"

for /l %%i in (8000,1,8010) do (
    if exist "%PID_DIR%\showdown_%%i.pid" (
        set /f "pid_line=" < "%PID_DIR%\showdown_%%i.pid"
        for /f "tokens=2 delims=:" %%p in ("!pid_line!") do (
            echo [INFO] Stopping server on port %%i ^(PID: %%p^)...
            taskkill /pid %%p /f >nul 2>&1
            if !errorlevel! equ 0 (
                echo   [OK] Server on port %%i stopped successfully
                del "%PID_DIR%\showdown_%%i.pid" >nul 2>&1
                set /a "stopped_count+=1"
            ) else (
                echo   [WARNING] Failed to stop PID %%p, may already be stopped
            )
        )
    ) else (
        REM Check if port is still in use
        netstat -an | findstr ":%%i " | findstr "LISTENING" >nul
        if !errorlevel! equ 0 (
            echo [INFO] Found server on port %%i without PID file, attempting to stop...
            for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%i " ^| findstr "LISTENING"') do (
                taskkill /pid %%a /f >nul 2>&1
                if !errorlevel! equ 0 (
                    echo   [OK] Server on port %%i stopped successfully
                    set /a "stopped_count+=1"
                )
            )
        )
    )
)

echo ===============================================
echo [SUCCESS] Successfully stopped %stopped_count% Pokemon Showdown server(s)

REM Additional cleanup - kill any remaining node processes running pokemon-showdown
echo.
echo [INFO] Checking for any remaining Pokemon Showdown processes...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq node.exe" /fo csv ^| findstr "pokemon-showdown"') do (
    taskkill /pid %%i /f >nul 2>&1
)

echo [INFO] All Pokemon Showdown servers have been stopped
goto :eof

:show_status
echo [INFO] Pokemon Showdown Server Status
echo Project root: %PROJECT_ROOT%
echo Checking port range: 8000-8010
echo ===============================================================================

REM Check Node.js version
echo System Information:
for /f "tokens=*" %%i in ('node --version 2^>nul') do echo   Node.js: %%i
echo   Current time: %date% %time%
echo.

echo Checking servers on ports 8000-8010...
echo -------------------------------------------------------------------------------

set "running_count=0"
set "managed_count=0"

for /l %%i in (8000,1,8010) do (
    netstat -an | findstr ":%%i " | findstr "LISTENING" >nul
    if !errorlevel! equ 0 (
        set /a "running_count+=1"
        
        REM Get PID from netstat
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%i " ^| findstr "LISTENING"') do (
            set "pid=%%a"
            
            REM Check if we have a PID file for this server
            if exist "%PID_DIR%\showdown_%%i.pid" (
                set /a "managed_count+=1"
                echo [RUNNING] Port %%i: RUNNING ^(Managed^)
                echo   PID: !pid!
                
                REM Check log file
                if exist "%LOGS_DIR%\showdown_server_%%i.log" (
                    for %%f in ("%LOGS_DIR%\showdown_server_%%i.log") do (
                        echo   Log: %LOGS_DIR%\showdown_server_%%i.log
                        echo   Log size: %%~zf bytes
                    )
                )
            ) else (
                echo [RUNNING] Port %%i: RUNNING ^(Unmanaged^)
                echo   PID: !pid!
            )
            echo.
        )
    ) else (
        echo [STOPPED] Port %%i: NOT RUNNING
        echo.
    )
)

echo ===============================================================================
echo.
echo Summary:
echo   Ports checked: 11
echo   Running servers: %running_count%
echo   Managed by script: %managed_count%
set /a "unmanaged_count=%running_count%-%managed_count%"
echo   Unmanaged processes: %unmanaged_count%
set /a "stopped_count=11-%running_count%"
echo   Stopped servers: %stopped_count%

echo.
echo Management commands:
echo   * Stop all servers: showdown.bat stop
echo   * Start servers: showdown.bat start [number]
echo   * View live logs: type "%LOGS_DIR%\showdown_server_[PORT].log"
echo.
echo Status check completed
goto :eof

:quick_start
echo [INFO] Quick start: Reading configuration from train_config.yml
if not exist "%CONFIG_FILE%" (
    echo [WARNING] Config file not found: %CONFIG_FILE%
    echo [INFO] Using default: 5 servers
    set "NUM_SERVERS=5"
) else (
    REM Try to extract parallel count from config file (simplified approach)
    findstr "parallel:" "%CONFIG_FILE%" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%i in ('findstr "parallel:" "%CONFIG_FILE%"') do (
            set "NUM_SERVERS=%%i"
        )
        echo [INFO] Found parallel configuration: !NUM_SERVERS!
    ) else (
        echo [INFO] No parallel configuration found, using default: 5 servers
        set "NUM_SERVERS=5"
    )
)

goto :start_servers

:show_help
echo.
echo Pokemon Showdown Server Management Script for Windows
echo.
echo Usage:
echo   showdown.bat start [number]    Start specified number of servers (default: 5)
echo   showdown.bat stop              Stop all Pokemon Showdown servers
echo   showdown.bat status            Show status of all servers
echo   showdown.bat quick             Auto-start based on train_config.yml
echo   showdown.bat help              Show this help message
echo.
echo Examples:
echo   showdown.bat start 5           Start 5 servers (ports 8000-8004)
echo   showdown.bat start 10          Start 10 servers (ports 8000-8009)
echo   showdown.bat stop              Stop all servers
echo   showdown.bat status            Check server status
echo   showdown.bat quick             Auto-start based on config
echo.
echo Notes:
echo   - Servers are started on ports 8000, 8001, 8002, etc.
echo   - Log files are saved to logs/showdown_logs/
echo   - PID files are saved to logs/pids/
echo   - Node.js must be installed and available in PATH
echo.
goto :eof