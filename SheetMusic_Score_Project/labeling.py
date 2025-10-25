import os
import shutil
import pandas as pd

def organize_wavs_by_csv(
    csv_path: str,
    audio_root: str,
    dry_run: bool = False
):
    """
    Organize wav files into class folders based on CSV metadata.

    Args:
        csv_path (str): Path to Metadata_Train.csv.
        audio_root (str): Path to folder containing all wav files.
        dry_run (bool): If True, only print planned moves.
    """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        fname = row['FileName']
        label = row['Class']
        src_path = os.path.join(audio_root, fname)
        dst_dir = os.path.join(audio_root, label)
        dst_path = os.path.join(dst_dir, fname)
        if not os.path.exists(src_path):
            print(f"Missing file: {src_path}")
            continue
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        if dry_run:
            print(f"Would move {src_path} -> {dst_path}")
        else:
            shutil.move(src_path, dst_path)
            print(f"Moved {src_path} -> {dst_path}")

if __name__ == "__main__":
    # Example usage
    organize_wavs_by_csv(
        csv_path="data/musical_instrument_sound/Metadata_Train.csv",
        audio_root="data/musical_instrument_sound",
        dry_run=False
    )