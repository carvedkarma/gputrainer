from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import MythosConfig
from .walkforward import run_mythos_walk_forward as _run_wf


def run_mythos_walk_forward(
    data_dir: Path,
    symbols: List[str],
    cfg: MythosConfig,
    report_path: Optional[Path] = None,
) -> Dict[str, object]:
    train_months = int(getattr(cfg, "train_months", 12))
    test_months = int(getattr(cfg, "test_months", 1))
    return _run_wf(
        data_dir=data_dir,
        symbols=symbols,
        train_months=train_months,
        test_months=test_months,
        config=cfg,
        output_path=report_path,
    )

