"""
TinySOL dataset downloader and subset sampler (≤1GB).
TODO: Implement Zenodo fetch, class filtering, and partial download based on size.
"""

from pathlib import Path
from typing import List, Optional

def download_tinysol(root: Path, classes: Optional[List[str]] = None, max_total_size_mb: int = 1000) -> Path:
    """Download TinySOL dataset (or a subset) under root and return local path.
    This is a stub; real implementation will handle partial download from Zenodo.
    """
    root = Path(root)
    (root / "tinysol").mkdir(parents=True, exist_ok=True)
    # TODO: implement actual download and filtering
    return root / "tinysol"
