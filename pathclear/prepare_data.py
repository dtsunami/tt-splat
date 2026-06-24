#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Real-data front end for 3DGS: accept EITHER a video or a folder of images,
run COLMAP (pycolmap, no sudo) -> camera poses + intrinsics + sparse points,
write a text model that train_real.py consumes.

  python prepare_data.py --video clip.mp4 --out runs/myscene --every 10
  python prepare_data.py --images photos/   --out runs/myscene

Video path needs ffmpeg (`sudo apt install ffmpeg`); the images path does not.
Output: <out>/sparse/0/{cameras,images,points3D}.txt  (+ <out>/images/)
"""
import argparse, os, shutil, subprocess, sys


def extract_frames(video, outdir, every):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found — `sudo apt install ffmpeg` (only needed for --video).")
    os.makedirs(outdir, exist_ok=True)
    # keep every Nth frame; robust to a 'bad' video (-err_detect ignore_err, no audio)
    subprocess.run(["ffmpeg", "-y", "-err_detect", "ignore_err", "-i", video,
                    "-vf", f"select=not(mod(n\\,{every}))", "-vsync", "0", "-an",
                    f"{outdir}/f%05d.png"], check=True)
    n = len([f for f in os.listdir(outdir) if f.endswith(".png")])
    print(f"extracted {n} frames -> {outdir}")
    if n < 8:
        print("WARNING: few frames — lower --every or the video is too short/bad for SfM.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--images"); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    import pycolmap

    work = a.out; os.makedirs(work, exist_ok=True)
    if a.video:
        imgdir = os.path.join(work, "images"); extract_frames(a.video, imgdir, a.every)
    elif a.images:
        imgdir = a.images
    else:
        sys.exit("need --video or --images")

    db = os.path.join(work, "db.db")
    if os.path.exists(db): os.remove(db)
    ro = pycolmap.ImageReaderOptions()
    try: ro.camera_model = "PINHOLE"        # match our pinhole renderer (ignore lens distortion for v1)
    except Exception: pass
    print("extracting features…")
    pycolmap.extract_features(db, imgdir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    print("matching…")
    pycolmap.match_exhaustive(db)
    sparse = os.path.join(work, "sparse"); os.makedirs(sparse, exist_ok=True)
    print("mapping (SfM)…")
    recs = pycolmap.incremental_mapping(db, imgdir, sparse)
    if not recs:
        sys.exit("COLMAP found no reconstruction — images lack overlap/texture (bad video?). "
                 "Try the image set, more frames, or a richer-texture capture.")
    rec = recs[0]
    out0 = os.path.join(sparse, "0")
    rec.write_text(out0)
    print(f"OK: {rec.num_images()} cameras, {rec.num_points3D()} points -> {out0}")
    print(f"next: python train_real.py --model {out0} --images {imgdir} --size 96")


if __name__ == "__main__":
    main()
