# ---------------------------------------------------------------------------
# 一键启动 backend（Windows PowerShell）
#   .\run_dev.ps1
# 然后浏览器打开 http://127.0.0.1:8000/
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "backend")

Write-Host "启动 backend ..." -ForegroundColor Cyan
Write-Host "  调试台   http://127.0.0.1:8000/" -ForegroundColor Green
Write-Host "  接口文档 http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "  按 Ctrl+C 停止" -ForegroundColor DarkGray
Write-Host ""

python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
