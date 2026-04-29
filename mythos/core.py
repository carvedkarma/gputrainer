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
    return _run_wf(
        data_dir=data_dir,
        symbols=symbols,
        train_months=cfg.train.train_months,
        test_months=cfg.train.test_months,
        config=cfg,
        output_path=report_path,
    )

