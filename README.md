# YOLO + SAHI Inference

This repo contains my trained YOLO model and a SAHI inference script.

## Files

- `best.pt` – trained YOLO model weights (39 MB)
- `sahi_scan.py` – script that runs sliced inference using SAHI

## How to use

1. Create a Python environment and install dependencies:
   pip install ultralytics sahi pillow
   Python 3.10.x recommended

3. Edit paths in the script and confidence thresholds for detection:
   (0.2 confidence threshold worked the best for me, not too conservative but not too many false positives)

