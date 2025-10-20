"""
IRMAS dataset downloader with class selection and ≤1GB subset.
TODO: Implement UPF/Zenodo fetching and partial sampling.
"""

from pathlib import Path
from typing import List, Optional

def download_irmas(root: Path, classes: Optional[List[str]] = None, max_files_per_class: int = 200, max_total_size_mb: int = 1000) -> Path:
    root = Path(root)
    (root / "irmas").mkdir(parents=True, exist_ok=True)
    # TODO: implement actual download, metadata parsing, and filtering
    return root / "irmas"
