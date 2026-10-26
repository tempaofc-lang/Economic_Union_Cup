# ---------------------------------------------------------------------------
# 一键启动 backend（Windows PowerShell）
#   .\run_dev.ps1
# 然后浏览器打开 http://127.0.0.1:8000/
#
# 注意 --reload-dir：
#   解译引擎在 手语识别系统/ 下（agent.py / pinyin_codec.py），不在 backend/ 内。
#   如果不额外声明，改 agent.py 不会触发热重载，会出现"改了却像没生效"的假象。
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
Set-Location (Join-Path $root "backend")

Write-Host "启动 backend ..." -ForegroundColor Cyan
Write-Host "  调试台   http://127.0.0.1:8000/" -ForegroundColor Green
Write-Host "  接口文档 http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  按 Ctrl+C 停止" -ForegroundColor DarkGray
Write-Host ""

python -m uvicorn main:app `
  --reload `
  --reload-dir . `
  --reload-dir (Join-Path $root "手语识别系统") `
  --host 127.0.0.1 `
  --port 8000
