# SentinelDMS

**Multimodal Driver Monitoring System: Fast CNN + Slow VLM**

SentinelDMS is a real-time driver monitoring prototype that combines a
high-frequency computer-vision safety loop with a slower vision-language
reasoning loop. The result is a cockpit-style DMS interface that can react to
classic fatigue signals while also explaining higher-level driver states such as
distraction, occlusion, abnormal behavior and cabin context.

<p align="center">
  <img src="SentinelDMS_images/cover_banner.png" alt="SentinelDMS banner" width="100%">
</p>

<p align="center">
  <b>30 FPS reflex path</b> for eyes, yawns and PERCLOS +
  <b>VLM reasoning path</b> for long-tail context and explainable alerts.
</p>

## What Changed

This repository started from a conventional YOLO + MediaPipe drowsiness detector
and was rebuilt into a dual-system driver monitoring stack:

- A custom PyQt cockpit UI with live camera HUD, animated risk gauge, telemetry
  cards, model status chips and natural-language AI briefing.
- A Fast System using MediaPipe Face Mesh, EAR, PERCLOS and YOLOv8 eye/yawn
  detectors for frame-by-frame physiological signals.
- A Slow System using Qwen/DashScope through an OpenAI-compatible API for
  multimodal driver-state reasoning.
- A confidence-weighted fusion layer that blends only the drowsiness dimension,
  while VLM-only dimensions pass directly to the cockpit.
- A startup sensor/model selection screen and mock mode for running the full UI
  without a cloud API key.

## System Architecture

<p align="center">
  <img src="SentinelDMS_images/architecture.png" alt="SentinelDMS dual-system architecture" width="100%">
</p>

The system is intentionally split into two paths:

| Path | Rate | Role | Core Signals |
|---|---:|---|---|
| Fast CNN System | ~30 FPS | Immediate reflex loop | EAR, PERCLOS, blinks, microsleeps, yawn count, yawn duration, fast confidence |
| Slow VLM System | latency-bound | Semantic reasoning loop | drowsiness, distraction, anomaly, occlusion, lighting, passengers, recommended action |
| Decision Fusion | every UI tick | Safety arbitration | confidence-weighted drowsiness score + direct VLM context |

The Fast System is optimized for stable real-time behavior. The Slow System is
optimized for situations that are hard to encode as single-purpose detectors:
masks, sunglasses, phone use, unusual posture, possible intoxication, emotional
distress, poor lighting and ambiguous facial visibility.

## Live Cockpit UI

<p align="center">
  <img src="SentinelDMS_images/hud_normal.png" alt="SentinelDMS normal monitoring state" width="100%">
</p>

The dashboard is not a generic demo window. It is built as an in-cabin operator
surface: a camera feed with HUD brackets, a fused risk gauge, fast vital signs,
slow contextual awareness and a natural-language AI briefing in one view.

<p align="center">
  <img src="SentinelDMS_images/hud_occlusion.png" alt="SentinelDMS occlusion-aware monitoring state" width="100%">
</p>

The VLM path adds semantic self-awareness. When the face is masked, occluded or
hard to judge, the system can state that reliability has changed instead of
silently producing a brittle vision-only decision.

## Fast System

The Fast System runs continuously on the latest webcam frame:

1. Detect face landmarks with MediaPipe Face Mesh.
2. Compute eye aspect ratio and sliding-window PERCLOS.
3. Crop eye and mouth regions from landmark geometry.
4. Run YOLOv8 eye/yawn detectors on the local ROIs.
5. Convert low-level signals into a fast drowsiness score and confidence.

This path is designed for low latency. It provides the reflex behavior needed
for a monitoring system: every frame updates the risk estimate even when the
VLM path is still thinking, offline or rate-limited.

## Slow System

The Slow System samples the latest frame in a background thread and sends a
compressed image to a VLM. It returns strict JSON with five driver-state
dimensions:

```json
{
  "drowsiness": {"level": 0, "confidence": 0.0},
  "distraction": {"detected": false, "type": "none", "confidence": 0.0},
  "anomaly": {"detected": false, "description": null, "severity": "none"},
  "occlusion": {"type": ["none"], "impact_on_reliability": 0.0},
  "context": {"lighting": "good", "passengers_detected": false},
  "overall_risk": 0,
  "explanation": "one paragraph driver-state analysis",
  "recommended_action": "none"
}
```

Only `drowsiness.level` is fused with the Fast System. The other fields are
separate VLM capabilities and are rendered directly in the UI.

## Fusion Logic

The fusion layer does not average everything. It uses the Fast System's own
confidence to decide how much the slow drowsiness estimate should matter:

| Fast confidence | Fast weight | Slow weight |
|---:|---:|---:|
| >= 0.80 | 0.80 | 0.20 |
| <= 0.50 | 0.20 | 0.80 |
| 0.50-0.80 | linear interpolation | linear interpolation |

If the VLM result is missing, stale or errored, SentinelDMS falls back to
Fast-only monitoring and keeps the UI alive.

## Engineering Pipeline

<p align="center">
  <img src="SentinelDMS_images/appendix_engineering_pipeline.png" alt="SentinelDMS engineering pipeline" width="100%">
</p>

The prototype connects a camera stream, MediaPipe/YOLO fast perception,
Qwen-based multimodal reasoning and a decision maker into one cockpit loop. The
current implementation focuses on local webcam operation, DashScope-compatible
VLM calls and a polished desktop UI suitable for live demos.

## Quick Start

```bash
git clone https://github.com/kfeng8021-spec/sentinel-dms.git
cd sentinel-dms

conda create -n sentinel-dms python=3.10 -y
conda activate sentinel-dms

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install openai

# Linux desktop note: if Qt/OpenCV plugin conflicts occur, use headless OpenCV.
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless opencv-contrib-python-headless
```

Run with a real Qwen/DashScope backend:

```bash
export DASHSCOPE_API_KEY=sk-your-key-here
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python DrowsinessDetector.py
```

Run in mock mode without a cloud API key:

```bash
unset DASHSCOPE_API_KEY
python DrowsinessDetector.py
```

The UI model selector can choose among supported Qwen/VL model names at startup.
If no API key is available, the Slow System produces mock structured outputs so
the full cockpit can still be exercised.

## Key Files

| File | Purpose |
|---|---|
| `DrowsinessDetector.py` | Main PyQt application, camera loop, fast metrics, model selector, HUD rendering and Fast/Slow integration |
| `slow_system.py` | Background VLM worker, prompt schema, DashScope/OpenAI-compatible request path and mock fallback |
| `decision_fusion.py` | Confidence-weighted drowsiness fusion logic |
| `runs/detecteye/train/weights/best.pt` | YOLOv8 eye-state detector weight used by the fast path |
| `runs/detectyawn/train/weights/best.pt` | YOLOv8 yawn detector weight used by the fast path |
| `SentinelDMS_images/` | README and business-plan visual assets |
| `NOTICE.md` | Upstream attribution and derivative-work notice |

## Roadmap

<p align="center">
  <img src="SentinelDMS_images/roadmap_en.png" alt="SentinelDMS roadmap" width="100%">
</p>

Planned development directions:

- Edge deployment on automotive SoC / NVIDIA Jetson-class hardware.
- IR camera support for low-light cabin monitoring.
- Audio event integration for yawning, snoring and speech-related distraction.
- Prompt profiles for Euro NCAP, C-NCAP and NHTSA-style monitoring protocols.
- Fleet dashboard integration for event review and OTA model updates.

## Project Scope

SentinelDMS is a research and demonstration prototype. It is not a certified
automotive safety product and should not be used as the sole basis for real
driving decisions.

The included model weights are demonstration weights. Production use would
require larger datasets, automotive-grade validation, privacy review, regulatory
assessment and edge-hardware profiling.

## Attribution

This repository is a derivative work based on
[`tyrerodr/real-time-drowsy-driving-detection`](https://github.com/tyrerodr/real-time-drowsy-driving-detection).
The original project provided the early YOLO/MediaPipe drowsiness-detection
foundation. SentinelDMS adds the VLM slow system, decision fusion layer, product
cockpit UI, model-selection flow, business-plan visuals and extended
documentation.

See [`NOTICE.md`](NOTICE.md) for the full attribution and takedown policy.
