#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/1 11:59
@Author  : adamzh0u
@File    : const.py
"""
from pathlib import Path
from loguru import logger

def get_project_root():
    """Search upwards to find the project root directory."""
    current_path = Path.cwd()
    while True:
        if (
            (current_path / ".git").exists()
            or (current_path / ".project_root").exists()
            or (current_path / ".gitignore").exists()
        ):
            logger.info(f"PROJECT_ROOT set to {str(current_path)}")
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            # loop until top level and land cwd
            cwd = Path.cwd()
            logger.info(f"PROJECT_ROOT set to current working directory: {str(cwd)}")
            return cwd
        current_path = parent_path

ROOT = get_project_root()
PATH_DATA = ROOT / "data"
PATH_NOOTBOOKS = ROOT / "notebooks"
PATH_LOG = ROOT / "logs"


