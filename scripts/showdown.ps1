# Pokemon Showdown Server Management Script for Windows PowerShell
# Usage: .\showdown.ps1 [start|stop|status|quick] [number_of_servers]

param(
    [Parameter(Position=0)]
    [string]$Command = "",
    
    [Parameter(Position=1)]
    [int]$NumServers = 5
)

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ShowdownDir = Join-Path $ProjectRoot "pokemon-showdown"
$LogsDir = Join-Path $ProjectRoot "logs\showdown_logs"
$PidDir = Join-Path $ProjectRoot "logs\pids"
$ConfigFile = Join-Path $ProjectRoot "config\train_config.yml"

# Create directories if they don't exist
if (!(Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null }
if (!(Test-Path $PidDir)) { New-Item -ItemType Directory -Path $PidDir -Force | Out-Null }

function Show-Help {
    Write-Host ""
    Write-Host "Pokemon Showdown Server Management Script for Windows PowerShell" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\showdown.ps1 start [number]    Start specified number of servers (default: 5)" -ForegroundColor White
    Write-Host "  .\showdown.ps1 stop              Stop all Pokemon Showdown servers" -ForegroundColor White
    Write-Host "  .\showdown.ps1 status            Show status of all servers" -ForegroundColor White
    Write-Host "  .\showdown.ps1 quick             Auto-start based on train_config.yml" -ForegroundColor White
    Write-Host "  .\showdown.ps1 help              Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\showdown.ps1 start 5           Start 5 servers (ports 8000-8004)" -ForegroundColor Gray
    Write-Host "  .\showdown.ps1 start 10          Start 10 servers (ports 8000-8009)" -ForegroundColor Gray
    Write-Host "  .\showdown.ps1 stop              Stop all servers" -ForegroundColor Gray
    Write-Host "  .\showdown.ps1 status            Check server status" -ForegroundColor Gray
    Write-Host "  .\showdown.ps1 quick             Auto-start based on config" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Notes:"
    Write-Host "  - Servers are started on ports 8000, 8001, 8002, etc."
    Write-Host "  - Log files are saved to logs/showdown_logs/"
    Write-Host "  - PID files are saved to logs/pids/"
    Write-Host "  - Node.js must be installed and available in PATH"
    Write-Host ""
}

function Start-Servers {
    param([int]$Count)
    
    Write-Host "🚀 Starting $Count Pokemon Showdown servers..." -ForegroundColor Green
    Write-Host "📂 Project root: $ProjectRoot"
    Write-Host "🔌 Port range: 8000-$(8000 + $Count - 1)"
    Write-Host ""
    
    # Check if Node.js is available
    try {
        $nodeVersion = & node --version 2>$null
        Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
        exit 1
    }
    
    # Check if pokemon-showdown directory exists
    if (!(Test-Path $ShowdownDir)) {
        Write-Host "❌ Pokemon Showdown directory not found: $ShowdownDir" -ForegroundColor Red
        Write-Host "❌ Please make sure Pokemon Showdown is properly installed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "🏁 Starting servers..."
    Write-Host "─────────────────────────────────────────────────────────────"
    
    $startedServers = 0
    
    for ($i = 0; $i -lt $Count; $i++) {
        $port = 8000 + $i
        $serverNum = $i + 1
        $logFile = Join-Path $LogsDir "showdown_server_$port.log"
        $pidFile = Join-Path $PidDir "showdown_$port.pid"
        
        Write-Host "🔧 Starting Pokemon Showdown server #$serverNum on port $port..." -ForegroundColor Cyan
        
        try {
            # Start server process
            $processInfo = New-Object System.Diagnostics.ProcessStartInfo
            $processInfo.FileName = "node"
            $processInfo.Arguments = "pokemon-showdown start --no-security --port $port"
            $processInfo.WorkingDirectory = $ShowdownDir
            $processInfo.UseShellExecute = $false
            $processInfo.RedirectStandardOutput = $true
            $processInfo.RedirectStandardError = $true
            $processInfo.CreateNoWindow = $true
            
            $process = New-Object System.Diagnostics.Process
            $process.StartInfo = $processInfo
            
            # Start process
            $process.Start() | Out-Null
            
            # Save PID
            "$port:$($process.Id)" | Out-File -FilePath $pidFile -Encoding UTF8
            
            # Start background job to handle logging
            Start-Job -ScriptBlock {
                param($Process, $LogFile)
                
                $outputReader = $Process.StandardOutput
                $errorReader = $Process.StandardError
                
                while (!$Process.HasExited) {
                    $output = $outputReader.ReadLine()
                    if ($output) {
                        Add-Content -Path $LogFile -Value $output
                    }
                    
                    $error = $errorReader.ReadLine()
                    if ($error) {
                        Add-Content -Path $LogFile -Value $error
                    }
                    
                    Start-Sleep -Milliseconds 100
                }
            } -ArgumentList $process, $logFile | Out-Null
            
            # Wait a moment for server to start
            Start-Sleep -Seconds 2
            
            # Check if server is running
            $listening = netstat -an | Select-String ":$port " | Select-String "LISTENING"
            if ($listening) {
                Write-Host "✅ Server #$serverNum started successfully (PID: $($process.Id), Port: $port)" -ForegroundColor Green
                Write-Host "   Log file: $logFile" -ForegroundColor Gray
                $startedServers++
            } else {
                Write-Host "❌ Server #$serverNum failed to start on port $port" -ForegroundColor Red
                if (Test-Path $pidFile) { Remove-Item $pidFile -Force }
            }
            
        } catch {
            Write-Host "❌ Failed to start server #$serverNum : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host "─────────────────────────────────────────────────────────────"
    Write-Host "🎉 Successfully started $startedServers Pokemon Showdown server(s)" -ForegroundColor Green
    Write-Host ""
    
    # Show server status
    Write-Host "📋 Server Status:"
    for ($i = 0; $i -lt $Count; $i++) {
        $port = 8000 + $i
        $serverNum = $i + 1
        
        $listening = netstat -an | Select-String ":$port " | Select-String "LISTENING"
        if ($listening) {
            Write-Host "   ✅ Server #$serverNum : http://localhost:$port - Running" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Server #$serverNum : http://localhost:$port - Failed" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "🔍 Management commands:"
    Write-Host "   • Check status: .\showdown.ps1 status"
    Write-Host "   • Stop all servers: .\showdown.ps1 stop"
    Write-Host "   • View logs: Get-Content `"$LogsDir\showdown_server_[PORT].log`" -Wait"
    Write-Host ""
    Write-Host "🚀 All servers are running in the background." -ForegroundColor Green
}

function Stop-Servers {
    Write-Host "🛑 Stopping Pokemon Showdown servers..." -ForegroundColor Yellow
    Write-Host "🔍 Checking port range: 8000-8010"
    Write-Host "─────────────────────────────────────────────────────────────"
    Write-Host ""
    
    $stoppedCount = 0
    
    # Stop servers using PID files
    for ($port = 8000; $port -le 8010; $port++) {
        $pidFile = Join-Path $PidDir "showdown_$port.pid"
        
        if (Test-Path $pidFile) {
            $pidContent = Get-Content $pidFile -Raw
            if ($pidContent -match ":(\d+)") {
                $pid = $Matches[1]
                Write-Host "🔄 Stopping server on port $port (PID: $pid)..." -ForegroundColor Cyan
                
                try {
                    Stop-Process -Id $pid -Force -ErrorAction Stop
                    Write-Host "   ✅ Server on port $port stopped successfully" -ForegroundColor Green
                    Remove-Item $pidFile -Force
                    $stoppedCount++
                } catch {
                    Write-Host "   ⚠️  Failed to stop PID $pid, may already be stopped" -ForegroundColor Yellow
                }
            }
        } else {
            # Check if port is still in use
            $listening = netstat -an | Select-String ":$port " | Select-String "LISTENING"
            if ($listening) {
                Write-Host "🔄 Found server on port $port without PID file, attempting to stop..." -ForegroundColor Cyan
                
                # Get PID from netstat
                $netstatOutput = netstat -ano | Select-String ":$port " | Select-String "LISTENING"
                if ($netstatOutput) {
                    $pid = ($netstatOutput -split '\s+')[-1]
                    try {
                        Stop-Process -Id $pid -Force -ErrorAction Stop
                        Write-Host "   ✅ Server on port $port stopped successfully" -ForegroundColor Green
                        $stoppedCount++
                    } catch {
                        Write-Host "   ⚠️  Failed to stop PID $pid" -ForegroundColor Yellow
                    }
                }
            }
        }
    }
    
    Write-Host "─────────────────────────────────────────────────────────────"
    Write-Host "✅ Successfully stopped $stoppedCount Pokemon Showdown server(s)" -ForegroundColor Green
    Write-Host ""
    
    # Additional cleanup
    Write-Host "🔍 Checking for any remaining Pokemon Showdown processes..."
    $nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq "node" -and $_.CommandLine -like "*pokemon-showdown*"
    }
    
    if ($nodeProcesses) {
        Write-Host "🧹 Cleaning up remaining processes..."
        $nodeProcesses | Stop-Process -Force
    }
    
    Write-Host "✨ All Pokemon Showdown servers have been stopped" -ForegroundColor Green
}

function Show-Status {
    Write-Host "🔍 Pokemon Showdown Server Status" -ForegroundColor Cyan
    Write-Host "📂 Project root: $ProjectRoot"
    Write-Host "🔌 Checking port range: 8000-8010"
    Write-Host "═══════════════════════════════════════════════════════════════════════════════"
    Write-Host ""
    
    # System information
    Write-Host "💻 System Information:"
    try {
        $nodeVersion = & node --version 2>$null
        Write-Host "   Node.js: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "   Node.js: Not found" -ForegroundColor Red
    }
    Write-Host "   Current time: $(Get-Date)"
    Write-Host ""
    
    Write-Host "🔍 Checking servers on ports 8000-8010..."
    Write-Host "─────────────────────────────────────────────────────────────"
    
    $runningCount = 0
    $managedCount = 0
    
    for ($port = 8000; $port -le 8010; $port++) {
        $listening = netstat -an | Select-String ":$port " | Select-String "LISTENING"
        
        if ($listening) {
            $runningCount++
            
            # Get PID from netstat
            $netstatOutput = netstat -ano | Select-String ":$port " | Select-String "LISTENING"
            $pid = ($netstatOutput -split '\s+')[-1]
            
            $pidFile = Join-Path $PidDir "showdown_$port.pid"
            if (Test-Path $pidFile) {
                $managedCount++
                Write-Host "🟢 Port $port : RUNNING (Managed)" -ForegroundColor Green
            } else {
                Write-Host "🟢 Port $port : RUNNING (Unmanaged)" -ForegroundColor Yellow
            }
            
            Write-Host "   PID: $pid" -ForegroundColor Gray
            
            # Check log file
            $logFile = Join-Path $LogsDir "showdown_server_$port.log"
            if (Test-Path $logFile) {
                $logSize = (Get-Item $logFile).Length
                $lastWrite = (Get-Item $logFile).LastWriteTime
                Write-Host "   Log: $logFile" -ForegroundColor Gray
                Write-Host "   Log size: $logSize bytes | Last activity: $lastWrite" -ForegroundColor Gray
            }
            Write-Host ""
        } else {
            Write-Host "⚪ Port $port : NOT RUNNING" -ForegroundColor Gray
            Write-Host ""
        }
    }
    
    Write-Host "═══════════════════════════════════════════════════════════════════════════════"
    Write-Host ""
    Write-Host "📊 Summary:"
    Write-Host "   🔍 Ports checked: 11"
    Write-Host "   🟢 Running servers: $runningCount"
    Write-Host "   📊 Managed by script: $managedCount"
    Write-Host "   ⚠️  Unmanaged processes: $($runningCount - $managedCount)"
    Write-Host "   🔴 Stopped servers: $(11 - $runningCount)"
    Write-Host ""
    
    # Process tree
    Write-Host "🌳 Pokemon Showdown Process Tree:"
    $nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue
    foreach ($proc in $nodeProcesses) {
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($commandLine -like "*pokemon-showdown*") {
                Write-Host "   📡 PID $($proc.Id): $commandLine" -ForegroundColor Cyan
            }
        } catch {
            # Ignore errors for processes we can't access
        }
    }
    Write-Host ""
    
    # Log files summary
    if (Test-Path $LogsDir) {
        $logFiles = Get-ChildItem $LogsDir -Filter "showdown_server_*.log"
        Write-Host "📋 Log Files Summary:"
        Write-Host "   Total log files: $($logFiles.Count)"
        $totalSize = ($logFiles | Measure-Object Length -Sum).Sum
        Write-Host "   Total log size: $totalSize bytes"
        Write-Host ""
        
        if ($logFiles.Count -gt 0) {
            Write-Host "   Recent log files:"
            $logFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
                Write-Host "   $($_.LastWriteTime) - $($_.Name) ($($_.Length) bytes)" -ForegroundColor Gray
            }
        }
    }
    
    Write-Host ""
    Write-Host "✅ $managedCount managed server(s) are running correctly" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔧 Management commands:"
    Write-Host "   • Stop all servers: .\showdown.ps1 stop"
    Write-Host "   • Start servers: .\showdown.ps1 start [number]"
    Write-Host "   • View live logs: Get-Content `"$LogsDir\showdown_server_[PORT].log`" -Wait"
    Write-Host ""
    Write-Host "🏁 Status check completed" -ForegroundColor Green
}

function Quick-Start {
    Write-Host "⚡ Quick start: Reading configuration from train_config.yml" -ForegroundColor Cyan
    
    if (!(Test-Path $ConfigFile)) {
        Write-Host "⚠️  Config file not found: $ConfigFile" -ForegroundColor Yellow
        Write-Host "ℹ️  Using default: 5 servers" -ForegroundColor Cyan
        $NumServers = 5
    } else {
        try {
            $configContent = Get-Content $ConfigFile -Raw
            if ($configContent -match "parallel:\s*(\d+)") {
                $NumServers = [int]$Matches[1]
                Write-Host "✅ Found parallel configuration: $NumServers" -ForegroundColor Green
            } else {
                Write-Host "ℹ️  No parallel configuration found, using default: 5 servers" -ForegroundColor Cyan
                $NumServers = 5
            }
        } catch {
            Write-Host "⚠️  Error reading config file, using default: 5 servers" -ForegroundColor Yellow
            $NumServers = 5
        }
    }
    
    Start-Servers -Count $NumServers
}

# Main script logic
switch ($Command.ToLower()) {
    "start" { Start-Servers -Count $NumServers }
    "stop" { Stop-Servers }
    "status" { Show-Status }
    "quick" { Quick-Start }
    "help" { Show-Help }
    "" { Show-Help; exit 1 }
    default { 
        Write-Host "❌ Unknown command: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}