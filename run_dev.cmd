@echo off
chcp 936 >nul 2>&1
setlocal
set "ROOT=%~dp0"
set "PORT=8000"

echo =====================================================
echo   手语视界 - backend 开发服务器
echo =====================================================
echo   调试台    http://127.0.0.1:%PORT%/
echo   接口文档  http://127.0.0.1:%PORT%/docs
echo   停止服务  按 Ctrl+C
echo.

cd /d "%ROOT%backend" 2>nul
if errorlevel 1 (
    echo [错误] 找不到 backend 目录
    goto :end
)

python -c "import fastapi,uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [错误] 当前 python 缺少 fastapi / uvicorn
    echo        请先执行: pip install -r backend\requirements.txt
    goto :end
)

rem 扫描含 agent.py 的目录，避免在批处理里写中文字面量
set "LEGACY="
for /d %%D in ("%ROOT%*") do if exist "%%D\agent.py" set "LEGACY=%%D"

if defined LEGACY (
    echo 热重载监听: backend + %LEGACY%
    echo.
    python -m uvicorn main:app --reload --host 127.0.0.1 --port %PORT% --reload-dir . --reload-dir "%LEGACY%"
) else (
    echo [提示] 没找到含 agent.py 的目录，只监听 backend
    echo.
    python -m uvicorn main:app --reload --host 127.0.0.1 --port %PORT%
)

:end
echo.
echo 服务已停止 / 启动失败，按任意键关闭...
pause >nul