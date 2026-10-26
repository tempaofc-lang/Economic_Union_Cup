"""
================================================================================
模块名称：backend/main.py
职责：FastAPI 入口，只做挂载（05 §2.2：main.py 不作业务逻辑）。

启动：
    cd backend
    python -m uvicorn main:app --reload --port 8000

然后打开：
    http://127.0.0.1:8000/          手动输入拼音的调试台
    http://127.0.0.1:8000/docs      自动生成的接口文档（可直接点着测）
================================================================================
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api import agent_api, meta, recognize
from config import APP_TITLE, APP_VERSION, FRONTEND_DIR

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "手语视界 backend。当前阶段：接入 DeepSeek 解译，"
        "并提供手动输入拼音的调试台，用于在 A 组模型未就绪时跑通全链路。"
    ),
)

# 本地联调放开跨域，方便李彤在别的端口起静态页面时直接 fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- API 路由 ----------------
app.include_router(meta.router, prefix="/api")
app.include_router(recognize.router, prefix="/api")
app.include_router(agent_api.router, prefix="/api")


# ---------------- 静态前端 ----------------
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        """调试台首页。"""
        return FileResponse(str(Path(FRONTEND_DIR) / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon = Path(FRONTEND_DIR) / "favicon.svg"
    if icon.exists():
        return FileResponse(str(icon), media_type="image/svg+xml")
    return FileResponse(str(Path(FRONTEND_DIR) / "index.html"))


if __name__ == "__main__":
    import uvicorn

    from config import HOST, PORT

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
