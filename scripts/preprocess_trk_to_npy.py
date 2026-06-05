import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from dipy.tracking.streamline import set_number_of_points
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Convert tractograms (.trk/.tck) to fixed-length .npy arrays")
    p.add_argument("--input-root", required=True, help="Root directory to scan")
    p.add_argument("--output-dir", required=True, help="Directory for output .npy files")
    p.add_argument("--pattern", default="*.trk", help="Glob pattern from input-root")
    p.add_argument("--num-points", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def subject_stem(path: Path) -> str:
    m = re.search(r"sub-(\d+)", path.name)
    if m:
        return f"tracts_{m.group(1)}"
    return path.stem


def convert_one(path_str: str, output_dir: str, num_points: int):
    import nibabel as nib

    path = Path(path_str)
    out = Path(output_dir) / f"{subject_stem(path)}.npy"

    trk = nib.streamlines.load(str(path))
    streamlines = trk.tractogram.streamlines
    if len(streamlines) == 0:
        return f"WARN empty: {path}"

    resampled = set_number_of_points(streamlines, num_points)
    arr = np.asarray(resampled, dtype=np.float32)
    np.save(out, arr)
    return f"OK {out.name}"


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{args.pattern}' under {input_root}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(convert_one, str(f), str(output_dir), args.num_points) for f in files]
        for fut in tqdm(futures, total=len(futures), desc="Converting"):
            fut.result()


if __name__ == "__main__":
    main()
