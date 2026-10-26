# =============================================================================
# 一键启动 backend（Windows PowerShell）
#     .\run_dev.ps1
#   然后浏览器打开 http://127.0.0.1:8000/
#
# 【为什么写这份脚本要格外小心】
#   双击 .ps1 时用的是 Windows PowerShell 5.1，它把「无 BOM 的 UTF-8 文件」按 ANSI 读取。
#   一旦脚本里出现中文字面量（比如直接写 手语识别系统），就会被解码成乱码，
#   拼出来的路径不存在，uvicorn 会报 "Invalid value for '--reload-dir'" 然后**立刻退出**，
#   表现就是窗口一闪而过（闪退）。
#
#   所以这里遵守两条纪律：
#     1) 代码里**不出现任何中文字面量**，目标目录靠扫描 agent.py 反查；
#     2) 本文件保存为 UTF-8 **带 BOM**，这样即使输出中文，5.1 也能正确显示。
# =============================================================================

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$port = 8000

function Pause-Exit([int]$code) {
    Write-Host ""
    Write-Host "按回车键关闭窗口..." -ForegroundColor DarkGray
    [void](Read-Host)
    exit $code
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  手语视界 · backend 开发服务器" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  调试台    http://127.0.0.1:$port/" -ForegroundColor Green
Write-Host "  接口文档  http://127.0.0.1:$port/docs" -ForegroundColor Green
Write-Host "  停止服务  按 Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

# ---- 1. 目录检查 ----
if (-not (Test-Path $backend)) {
    Write-Host "[错误] 找不到 backend 目录：$backend" -ForegroundColor Red
    Pause-Exit 1
}

# ---- 2. 依赖检查（缺依赖直接说清楚，别让它闪退） ----
$depOk = $false
try {
    python -c "import fastapi, uvicorn" 2>$null
    $depOk = ($LASTEXITCODE -eq 0)
} catch {
    $depOk = $false
}

if (-not $depOk) {
    Write-Host "[错误] 当前 python 缺少 fastapi / uvicorn。" -ForegroundColor Red
    Write-Host "       请先执行：pip install -r backend\requirements.txt" -ForegroundColor Yellow
    Pause-Exit 1
}

# ---- 3. 端口占用检查 ----
$busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $owner = ($busy | Select-Object -First 1).OwningProcess
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue
    Write-Host "[错误] 端口 $port 已被占用（PID $owner）。" -ForegroundColor Red
    if ($proc) { Write-Host "       $($proc.CommandLine)" -ForegroundColor DarkGray }
    Write-Host "       很可能是上一次的服务没关。解决办法：" -ForegroundColor Yellow
    Write-Host "         ① 回到那个窗口按 Ctrl+C；或" -ForegroundColor Yellow
    Write-Host "         ② 执行：Stop-Process -Id $owner -Force" -ForegroundColor Yellow
    Pause-Exit 1
}

# ---- 4. 定位解译引擎目录（扫描 agent.py，避免中文字面量） ----
$legacy = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "agent.py") } |
    Select-Object -First 1

Set-Location $backend

# 用参数数组而不是反引号续行：反引号对行尾空格敏感，很容易踩坑
$uvArgs = @("-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "$port")
if ($legacy) {
    $uvArgs += @("--reload-dir", ".", "--reload-dir", $legacy.FullName)
    Write-Host "热重载监听：backend + $($legacy.Name)" -ForegroundColor DarkGray
} else {
    Write-Host "[提示] 没找到包含 agent.py 的目录，将只监听 backend（改 agent.py 不会热重载）" -ForegroundColor Yellow
}
Write-Host ""

try {
    & python @uvArgs
} catch {
    Write-Host ""
    Write-Host "[错误] 启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Pause-Exit 1
}

# uvicorn 正常退出（Ctrl+C）时也停留一下，避免窗口直接消失
Write-Host ""
Write-Host "服务已停止。" -ForegroundColor DarkGray
Pause-Exit 0
