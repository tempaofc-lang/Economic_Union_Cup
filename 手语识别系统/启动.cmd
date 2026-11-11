@echo off
chcp 65001 >nul
title 手语视界 - 录制版 V13

echo ============================================================
echo   手语视界 · 录制版 V13  一键启动
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] 关闭所有残留的 Python 进程...
taskkill /F /IM python.exe >nul 2>&1
if errorlevel 1 (
    echo       没有残留进程，继续。
) else (
    echo       已清理旧进程。
)
timeout /t 2 >nul

echo.
echo [2/4] 校验当前代码版本...
findstr /C:"V13" app.py >nul 2>&1
if errorlevel 1 (
    echo.
    echo   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo   ！ 错误：当前目录的 app.py 不是 V13 版本！
    echo   ！ 请把下载的 app.py 复制到：
    echo   ！      %cd%
    echo   ！ 并选择"替换目标中的文件"
    echo   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo.
    pause
    exit /b 1
)
echo       app.py 版本校验通过。

echo.
echo [3/4] 设置本地代理绕过...
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

echo.
echo [4/4] 启动服务（浏览器会自动打开）...
echo ------------------------------------------------------------
echo   重要：看到 [record] 字样说明按钮点击已被接收
echo   若点【开始识别】后没有 [record]，说明页面是旧的，
echo   请按 Ctrl+F5 强制刷新浏览器。
echo ------------------------------------------------------------
echo.

start "" cmd /c "timeout /t 12 >nul && start http://127.0.0.1:7860"

python app.py

echo.
echo 服务已停止。
pause
