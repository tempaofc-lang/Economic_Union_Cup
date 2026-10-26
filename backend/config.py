"""
================================================================================
模块名称：backend/config.py
职责：集中管理路径与运行配置，避免各模块到处拼路径。
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/ 目录
BACKEND_DIR = Path(__file__).resolve().parent
# 项目根目录（Economic_Union_Cup/）
PROJECT_ROOT = BACKEND_DIR.parent
# 现有的解译引擎与词表所在目录（阶段性复用，后续按 05 §2.2 迁入 backend/）
LEGACY_DIR = PROJECT_ROOT / "手语识别系统"

DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIR = BACKEND_DIR / "frontend"
GOV_KB_PATH = DATA_DIR / "gov_kb.json"

HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
PORT = int(os.getenv("BACKEND_PORT", "8000"))

APP_TITLE = "手语视界 · 政务服务无障碍导办 Agent"
APP_VERSION = "0.1.0"
