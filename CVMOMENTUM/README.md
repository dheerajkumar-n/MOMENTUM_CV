# MOMENTUM ⚡
### Motion-Oriented Multi-Criteria Evaluation for Natural Temporal Understanding of Moments

A computer vision system that analyzes dynamic videos (dance, sports, performance) and automatically extracts the most visually meaningful peak moment — then enhances it to be post-worthy.

---

## What it does

1. **Temporal Motion Analysis** — Dense optical flow (OpenCV) tracks motion energy across every frame
2. **Pose-Based Peak Detection** — MediaPipe Pose scores frames by limb extension, body spread, and pose expressiveness
3. **Face Clarity Scoring** — MediaPipe FaceMesh rewards frames where faces are sharp and well-visible
4. **Multi-Criteria Ranking** — Composite score (motion 30% + pose 30% + face 25% + sharpness 15%) ranks all frames with temporal diversity enforcement
5. **Vision-Based Enhancement (ART)** — Subject-aware smart crop → CLAHE exposure correction → Vibrance boost → Unsharp masking → Cinematic color grade (warm shadows / cool highlights) → Vignette

**Output:** Top 3 peak frames × pre-edit + post-edit = 6 downloadable images

---

## Setup

```bash
# Clone / copy the project folder
cd momentum

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. Upload any video (MP4, MOV, AVI, MKV, WEBM)
2. Click **ANALYZE**
3. View the temporal motion graph with detected peak moments marked
4. See Top 3 frames with score breakdowns (before enhancement)
5. See the same 3 frames post-enhancement with cinematic processing applied
6. Download any frame

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Optical Flow | OpenCV (Farneback) |
| Pose Detection | MediaPipe Pose |
| Face Detection | MediaPipe FaceMesh |
| Image Enhancement | OpenCV, scikit-image |
| UI | Streamlit |
| Signal Processing | SciPy, NumPy |
| Visualization | Matplotlib |

---

## Team
- **Dheeraj** — Temporal Motion Analysis, Pose-Based Peak Moment Detection
- **Yash** — Image Quality Scoring, Vision-Based Aesthetic Enhancement

GSU Computer Vision Project
