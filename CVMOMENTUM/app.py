import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pipeline import (
    analyze_video,
    get_top_frames,
    enhance_frame,
    frames_to_motion_data,
    FrameScore
)

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOMENTUM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLES ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a24;
    --accent: #c084fc;
    --accent2: #818cf8;
    --accent3: #f472b6;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: rgba(192, 132, 252, 0.15);
  }

  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  .stApp {
    background: var(--bg) !important;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem !important; max-width: 1400px; }

  /* Hero header */
  .momentum-hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
  }

  .momentum-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(3rem, 8vw, 7rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 40%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
  }

  .momentum-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.75rem;
  }

  .momentum-tagline {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
  }

  /* Upload zone */
  .upload-zone {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    background: var(--surface);
    text-align: center;
    margin: 2rem auto;
    max-width: 600px;
    transition: border-color 0.3s;
  }

  .upload-zone:hover {
    border-color: rgba(192, 132, 252, 0.4);
  }

  /* Section headers */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Frame cards */
  .frame-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s, transform 0.2s;
  }

  .frame-card:hover {
    border-color: rgba(192, 132, 252, 0.35);
    transform: translateY(-2px);
  }

  .frame-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
  }

  .frame-timestamp {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.5rem;
  }

  /* Score bar */
  .score-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 0.3rem 0;
    font-size: 0.78rem;
  }

  .score-label {
    color: var(--muted);
    width: 70px;
    font-size: 0.72rem;
  }

  .score-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--surface2);
    border-radius: 2px;
    margin: 0 0.75rem;
    overflow: hidden;
  }

  .score-bar-fill {
    height: 100%;
    border-radius: 2px;
  }

  .score-value {
    color: var(--text);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    width: 35px;
    text-align: right;
  }

  /* Stats chips */
  .stat-chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    margin: 0.2rem;
  }

  /* Before/After divider */
  .ba-divider {
    text-align: center;
    padding: 2rem 0 1rem;
    position: relative;
  }

  .ba-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-transform: uppercase;
    background: var(--bg);
    padding: 0 1rem;
    position: relative;
    z-index: 1;
  }

  /* Progress */
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #c084fc, #818cf8) !important;
  }

  /* File uploader override */
  [data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
  }

  [data-testid="stFileUploader"]:hover {
    border-color: rgba(192, 132, 252, 0.4) !important;
  }

  /* Download button */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #c084fc, #818cf8) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.25rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
  }

  .stDownloadButton > button:hover {
    opacity: 0.85 !important;
  }

  /* Spinner */
  .stSpinner > div {
    border-top-color: var(--accent) !important;
  }

  /* Plot background */
  .plot-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
  }

  /* Alert/info */
  .stAlert {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  /* Columns gap */
  [data-testid="column"] { padding: 0 0.5rem; }

  /* Footer */
  .momentum-footer {
    text-align: center;
    padding: 3rem 0 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
  }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────

def frame_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def frame_to_bytes(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()

def score_bar_html(label: str, value: float, color: str) -> str:
    pct = int(value * 100)
    return f"""
    <div class="score-row">
      <span class="score-label">{label}</span>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
      <span class="score-value">{pct}%</span>
    </div>
    """

def make_motion_plot(timestamps, motions, composites, peak_times):
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='#111118')
    ax.set_facecolor('#111118')

    ax.fill_between(timestamps, motions, alpha=0.2, color='#818cf8')
    ax.plot(timestamps, motions, color='#818cf8', linewidth=1.2, label='Motion Energy', alpha=0.8)
    ax.plot(timestamps, composites, color='#c084fc', linewidth=1.8, label='Composite Score')

    for pt in peak_times:
        ax.axvline(x=pt, color='#f472b6', linewidth=1.5, alpha=0.8, linestyle='--')

    ax.set_xlabel('Time (s)', color='#64748b', fontsize=9)
    ax.set_ylabel('Score', color='#64748b', fontsize=9)
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1a1a24')
    ax.legend(fontsize=8, labelcolor='#94a3b8',
              facecolor='#1a1a24', edgecolor='#1a1a24',
              loc='upper right')
    ax.grid(axis='y', color='#1a1a24', linewidth=0.5)

    plt.tight_layout()
    return fig


# ─── MAIN APP ──────────────────────────────────────────────────────────────

def main():
    # Hero
    st.markdown("""
    <div class="momentum-hero">
      <h1 class="momentum-title">MOMENTUM</h1>
      <p class="momentum-sub">Motion-Oriented Multi-Criteria Evaluation for Natural Temporal Understanding of Moments</p>
      <p class="momentum-tagline">Drop a video. Get the perfect frame.</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    col_l, col_upload, col_r = st.columns([1, 2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility="collapsed",
            help="Dance, sports, performance — any dynamic video works best"
        )
        st.markdown('<p style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:0.5rem;">MP4 · MOV · AVI · MKV · WEBM</p>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 0; color: #334155;">
          <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
          <p style="font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase;">
            Upload a video to begin analysis
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="momentum-footer">MOMENTUM · CV PROJECT · YASH & DHEERAJ</div>', unsafe_allow_html=True)
        return

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Video info
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    col1, col2, col3, col4 = st.columns(4)
    chip_style = "text-align:center; background:#111118; border:1px solid rgba(192,132,252,0.15); border-radius:10px; padding:0.75rem; margin:0.25rem;"
    val_style = "font-family:'Space Mono',monospace; font-size:1.1rem; color:#c084fc; display:block;"
    lbl_style = "font-size:0.68rem; color:#64748b; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.2rem; display:block;"

    with col1:
        st.markdown(f'<div style="{chip_style}"><span style="{val_style}">{duration:.1f}s</span><span style="{lbl_style}">Duration</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="{chip_style}"><span style="{val_style}">{fps:.0f}</span><span style="{lbl_style}">FPS</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="{chip_style}"><span style="{val_style}">{total_frames}</span><span style="{lbl_style}">Frames</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div style="{chip_style}"><span style="{val_style}">{width}×{height}</span><span style="{lbl_style}">Resolution</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button
    if 'results' not in st.session_state or st.session_state.get('last_file') != uploaded.name:
        if st.button("⚡  ANALYZE", use_container_width=True, type="primary"):
            with st.spinner("Analyzing video..."):
                progress_bar = st.progress(0, text="Reading frames...")
                status = st.empty()

                def progress_cb(p):
                    progress_bar.progress(min(p, 0.95), text=f"Analyzing frames… {int(p*100)}%")

                status.markdown('<p style="color:#64748b; font-size:0.8rem; text-align:center;">Running pose detection, optical flow & face analysis…</p>', unsafe_allow_html=True)
                t0 = time.time()
                all_scores = analyze_video(tmp_path, progress_callback=progress_cb)
                elapsed = time.time() - t0

                progress_bar.progress(1.0, text="Done!")
                status.empty()

                if not all_scores:
                    st.error("Could not analyze video. Make sure the video contains visible subjects.")
                    return

                top3 = get_top_frames(all_scores, n=3)
                enhanced = [enhance_frame(f.frame) for f in top3]

                st.session_state['results'] = {
                    'all_scores': all_scores,
                    'top3': top3,
                    'enhanced': enhanced,
                    'elapsed': elapsed
                }
                st.session_state['last_file'] = uploaded.name
                st.rerun()

    # ─── RESULTS ───────────────────────────────────────────────────────────
    if 'results' in st.session_state and st.session_state.get('last_file') == uploaded.name:
        res = st.session_state['results']
        all_scores = res['all_scores']
        top3 = res['top3']
        enhanced = res['enhanced']
        elapsed = res['elapsed']

        st.markdown(f'<p style="text-align:center; color:#64748b; font-size:0.75rem; font-family:Space Mono,monospace; letter-spacing:0.1em;">Analysis complete in {elapsed:.1f}s · {len(all_scores)} frames sampled</p>', unsafe_allow_html=True)

        # ── Motion graph ──
        st.markdown('<div class="section-label">Temporal Analysis</div>', unsafe_allow_html=True)
        timestamps, motions, composites = frames_to_motion_data(all_scores)
        peak_times = [f.timestamp for f in top3]
        fig = make_motion_plot(timestamps, motions, composites, peak_times)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PRE-EDIT frames ──
        st.markdown('<div class="section-label">Peak Moments — Original</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        rank_labels = ["#01 — BEST", "#02 — SECOND", "#03 — THIRD"]
        accent_colors = ["#c084fc", "#818cf8", "#f472b6"]

        for i, (col, frame_score) in enumerate(zip(cols, top3)):
            with col:
                pil_img = frame_to_pil(frame_score.frame)
                st.image(pil_img, use_container_width=True)

                st.markdown(f"""
                <div class="frame-card">
                  <div class="frame-rank" style="color:{accent_colors[i]};">{rank_labels[i]}</div>
                  {score_bar_html("Motion", frame_score.motion_score, accent_colors[i])}
                  {score_bar_html("Pose", frame_score.pose_score, accent_colors[i])}
                  {score_bar_html("Face", frame_score.face_score, accent_colors[i])}
                  {score_bar_html("Sharp", frame_score.sharpness_score, accent_colors[i])}
                  <div style="border-top:1px solid rgba(255,255,255,0.05); margin:0.6rem 0 0.4rem;"></div>
                  {score_bar_html("Total", frame_score.composite_score, "#ffffff")}
                  <div class="frame-timestamp">⏱ {frame_score.timestamp:.2f}s · frame {frame_score.frame_idx}</div>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label=f"↓ DOWNLOAD ORIGINAL",
                    data=frame_to_bytes(frame_score.frame),
                    file_name=f"momentum_original_{i+1}_{frame_score.timestamp:.2f}s.jpg",
                    mime="image/jpeg",
                    key=f"dl_orig_{i}",
                    use_container_width=True
                )

        # ── POST-EDIT frames ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
          <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(192,132,252,0.3), rgba(129,140,248,0.3), transparent); margin-bottom:1rem;"></div>
          <span style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.35em; color:#64748b; text-transform:uppercase;">
            ✦ Vision-Based Enhancement Applied ✦
          </span>
          <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(192,132,252,0.3), rgba(129,140,248,0.3), transparent); margin-top:1rem;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-label">Peak Moments — Enhanced</div>', unsafe_allow_html=True)

        cols2 = st.columns(3)
        for i, (col, enh_frame, orig_score) in enumerate(zip(cols2, enhanced, top3)):
            with col:
                pil_enh = frame_to_pil(enh_frame)
                st.image(pil_enh, use_container_width=True)

                # Enhancement metrics
                orig_sharp = orig_score.sharpness_score
                enh_sharp_raw = cv2.Laplacian(cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                enh_sharp_norm = min(enh_sharp_raw / (cv2.Laplacian(cv2.cvtColor(orig_score.frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() + 1e-6), 2.0) / 2.0

                st.markdown(f"""
                <div class="frame-card" style="border-color:rgba({
                    '192,132,252' if i==0 else ('129,140,248' if i==1 else '244,114,182')
                }, 0.3);">
                  <div class="frame-rank" style="color:{accent_colors[i]};">ENHANCED · {rank_labels[i]}</div>
                  <div style="display:flex; flex-wrap:wrap; gap:0.3rem; margin-top:0.4rem;">
                    <span style="background:#1a1a24; border:1px solid rgba(192,132,252,0.15); border-radius:20px; padding:0.2rem 0.6rem; font-size:0.65rem; font-family:'Space Mono',monospace; color:#c084fc;">CLAHE</span>
                    <span style="background:#1a1a24; border:1px solid rgba(192,132,252,0.15); border-radius:20px; padding:0.2rem 0.6rem; font-size:0.65rem; font-family:'Space Mono',monospace; color:#818cf8;">VIBRANCE</span>
                    <span style="background:#1a1a24; border:1px solid rgba(192,132,252,0.15); border-radius:20px; padding:0.2rem 0.6rem; font-size:0.65rem; font-family:'Space Mono',monospace; color:#f472b6;">GRADE</span>
                    <span style="background:#1a1a24; border:1px solid rgba(192,132,252,0.15); border-radius:20px; padding:0.2rem 0.6rem; font-size:0.65rem; font-family:'Space Mono',monospace; color:#94a3b8;">VIGNETTE</span>
                  </div>
                  <div class="frame-timestamp" style="margin-top:0.6rem;">⏱ {orig_score.timestamp:.2f}s — post-processed</div>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label=f"↓ DOWNLOAD ENHANCED",
                    data=frame_to_bytes(enh_frame),
                    file_name=f"momentum_enhanced_{i+1}_{orig_score.timestamp:.2f}s.jpg",
                    mime="image/jpeg",
                    key=f"dl_enh_{i}",
                    use_container_width=True
                )

        # ── Re-analyze button ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  ANALYZE NEW VIDEO", use_container_width=False):
            for key in ['results', 'last_file']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    st.markdown('<div class="momentum-footer">MOMENTUM · Computer Vision · Yash & Dheeraj · GSU</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
