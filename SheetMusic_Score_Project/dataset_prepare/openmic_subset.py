"""
OpenMIC-2018 subset downloader using metadata; limit to ≤1GB.
TODO: Fetch from Zenodo, filter by selected classes and sample count.
"""

from pathlib import Path
from typing import List, Optional

def download_openmic_subset(root: Path, classes: Optional[List[str]] = None, max_files_per_class: int = 150, max_total_size_mb: int = 1000) -> Path:
    root = Path(root)
    (root / "openmic").mkdir(parents=True, exist_ok=True)
    # TODO: implement metadata-based subset fetch and copy
    return root / "openmic"
