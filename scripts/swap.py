#!/usr/bin/env python3
"""Face-swap one video: put source face onto every frame, keep original audio.
Usage: swap.py <source_img> <in_video> <out_video> [model_path]
"""
import sys, os, subprocess
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def largest(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def main():
    src_path, in_video, out_video = sys.argv[1], sys.argv[2], sys.argv[3]
    model_path = sys.argv[4] if len(sys.argv) > 4 else os.path.abspath("inswapper_128.onnx")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(model_path, providers=["CPUExecutionProvider"])

    src_img = cv2.imread(src_path)
    if src_img is None:
        raise SystemExit(f"cannot read source image {src_path}")
    src_face = largest(app.get(src_img))
    if src_face is None:
        raise SystemExit("no face found in source image")

    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {in_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp = out_video + ".noaudio.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    i = 0
    swapped_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        if faces:
            tgt = largest(faces)
            frame = swapper.get(frame, tgt, src_face, paste_back=True)
            swapped_frames += 1
        writer.write(frame)
        i += 1
        if i % 50 == 0:
            print(f"  {i}/{n} frames", flush=True)
    cap.release()
    writer.release()
    print(f"  done {i} frames, face swapped on {swapped_frames}", flush=True)

    # re-encode to web-friendly h264 and mux the ORIGINAL audio back
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp, "-i", in_video,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
        "-c:a", "aac", "-shortest", out_video,
    ], check=True)
    os.remove(tmp)
    print(f"  wrote {out_video}", flush=True)

if __name__ == "__main__":
    main()
