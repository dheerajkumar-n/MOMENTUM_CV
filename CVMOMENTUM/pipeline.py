import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
from skimage import exposure, filters
import tempfile
import os
from dataclasses import dataclass
from typing import List, Tuple

# Frames are downscaled to this width before ALL inference (pose, face, flow,
# sharpness).  Landmark coords are normalised [0→1] so scores are unaffected.
# 640 px wide ≈ 36× fewer pixels than 4K → large speed-up.
INFERENCE_WIDTH = 640


def _inference_resize(frame: np.ndarray) -> np.ndarray:
    """Return frame resized to INFERENCE_WIDTH, keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= INFERENCE_WIDTH:
        return frame
    new_h = int(h * INFERENCE_WIDTH / w)
    return cv2.resize(frame, (INFERENCE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

# ─── MediaPipe Tasks model paths ───────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(_DIR, "pose_landmarker_full.task")
FACE_MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")

def _ensure_models():
    """Download MediaPipe Tasks model bundles if not already present."""
    if not os.path.exists(POSE_MODEL_PATH):
        print("Downloading pose landmarker model (~25 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            POSE_MODEL_PATH,
        )
        print("Pose model downloaded.")
    if not os.path.exists(FACE_MODEL_PATH):
        print("Downloading face landmarker model (~6 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task",
            FACE_MODEL_PATH,
        )
        print("Face model downloaded.")


@dataclass
class FrameScore:
    frame_idx: int
    timestamp: float
    motion_score: float
    pose_score: float
    face_score: float
    sharpness_score: float
    composite_score: float
    frame: np.ndarray


def compute_optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute motion energy between two frames using dense optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def compute_pose_score(frame: np.ndarray, pose_result) -> float:
    """
    Score a frame based on pose expressiveness.
    Higher score = more extended/dynamic pose (peak action).
    Accepts a PoseLandmarkerResult from the Tasks API.
    """
    if not pose_result.pose_landmarks:
        return 0.0

    landmarks = pose_result.pose_landmarks[0]  # first detected person
    h, w = frame.shape[:2]

    # Extract key joint positions
    def lm(idx):
        l = landmarks[idx]
        return np.array([l.x * w, l.y * h])

    # MediaPipe pose landmark indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28

    try:
        # Arm spread: distance between wrists
        wrist_span = np.linalg.norm(lm(LEFT_WRIST) - lm(RIGHT_WRIST))

        # Leg spread: distance between ankles
        ankle_span = np.linalg.norm(lm(LEFT_ANKLE) - lm(RIGHT_ANKLE))

        # Body height in frame (shoulder to ankle midpoint)
        shoulder_mid = (lm(LEFT_SHOULDER) + lm(RIGHT_SHOULDER)) / 2
        ankle_mid = (lm(LEFT_ANKLE) + lm(RIGHT_ANKLE)) / 2
        body_height = np.linalg.norm(shoulder_mid - ankle_mid)

        # Elbow angle (left) - more acute = more bent = less extended
        def angle_between(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        left_elbow_angle = angle_between(lm(LEFT_SHOULDER), lm(LEFT_ELBOW), lm(LEFT_WRIST))
        right_elbow_angle = angle_between(lm(RIGHT_SHOULDER), lm(RIGHT_ELBOW), lm(RIGHT_WRIST))
        left_knee_angle = angle_between(lm(LEFT_HIP), lm(LEFT_KNEE), lm(LEFT_ANKLE))
        right_knee_angle = angle_between(lm(RIGHT_HIP), lm(RIGHT_KNEE), lm(RIGHT_ANKLE))

        # Score components
        # Normalize by frame diagonal
        diag = np.sqrt(w**2 + h**2)
        span_score = (wrist_span + ankle_span) / diag

        # Extension score: straighter limbs = higher score (more dramatic pose)
        extension_score = (left_elbow_angle + right_elbow_angle + left_knee_angle + right_knee_angle) / (4 * 180)

        # Vertical height score (person filling frame)
        height_score = body_height / h

        # Visibility score: average landmark visibility
        vis_score = np.mean([l.visibility for l in landmarks])

        composite = (0.3 * span_score + 0.3 * extension_score + 0.2 * height_score + 0.2 * vis_score)
        return float(composite)

    except Exception:
        return 0.0


def compute_face_score(frame: np.ndarray, face_result) -> float:
    """
    Score faces in frame: reward clarity, penalize blur/occlusion.
    Accepts a FaceLandmarkerResult from the Tasks API.
    """
    if not face_result.face_landmarks:
        return 0.0

    scores = []
    h, w = frame.shape[:2]

    for face_lms in face_result.face_landmarks:
        # Estimate face bounding box
        xs = [lm.x * w for lm in face_lms]
        ys = [lm.y * h for lm in face_lms]
        x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
        x2, y2 = int(min(w, max(xs))), int(min(h, max(ys)))

        if x2 <= x1 or y2 <= y1:
            continue

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Sharpness of face region
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

        # Face size relative to frame
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        size_score = face_area / frame_area

        face_score = 0.6 * min(sharpness / 500, 1.0) + 0.4 * min(size_score * 10, 1.0)
        scores.append(face_score)

    return float(np.mean(scores)) if scores else 0.0


def compute_sharpness(frame: np.ndarray) -> float:
    """Overall frame sharpness via Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze_video(video_path: str, progress_callback=None) -> List[FrameScore]:
    """
    Full temporal analysis of video. Returns scored frames.
    """
    _ensure_models()

    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_options)
    face_detector = mp_vision.FaceLandmarker.create_from_options(face_options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Aim for ~150 candidate frames — still 5+ samples/sec for a 30 s clip,
    # which is far more than needed given the 1-second diversity window.
    sample_interval = max(1, total_frames // 150)

    scores = []
    prev_gray = None
    frame_idx = 0
    last_timestamp_ms = -1

    # Two-thread executor: pose and face run in parallel per sampled frame.
    with ThreadPoolExecutor(max_workers=2) as pool:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # ── Downscale for all inference ──────────────────────────────
                small = _inference_resize(frame)
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                # Motion score (on downscaled grays — relative, so normalises fine)
                motion = compute_optical_flow_score(prev_gray, gray) if prev_gray is not None else 0.0

                # Build shared MediaPipe image + monotonic timestamp
                rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms      = max(int(frame_idx * 1000 / fps), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                # Pose + face in parallel (independent detectors → safe)
                pose_fut = pool.submit(pose_detector.detect_for_video, mp_image, timestamp_ms)
                face_fut = pool.submit(face_detector.detect_for_video, mp_image, timestamp_ms)
                pose_s   = compute_pose_score(small, pose_fut.result())
                face_s   = compute_face_score(small, face_fut.result())

                # Sharpness on downscaled frame (relative ranking is preserved)
                sharp = compute_sharpness(small)

                scores.append(FrameScore(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    motion_score=motion,
                    pose_score=pose_s,
                    face_score=face_s,
                    sharpness_score=sharp,
                    composite_score=0.0,  # computed after normalisation
                    frame=frame.copy()    # store original full-res for enhancement
                ))

                prev_gray = gray

                if progress_callback:
                    progress_callback(frame_idx / max(total_frames, 1))

            frame_idx += 1

    cap.release()
    pose_detector.close()
    face_detector.close()

    if not scores:
        return []

    # Normalize each score component to [0, 1]
    def norm(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    motions = norm([s.motion_score for s in scores])
    poses = norm([s.pose_score for s in scores])
    faces = norm([s.face_score for s in scores])
    sharps = norm([s.sharpness_score for s in scores])

    # Weighted composite
    W_MOTION = 0.30
    W_POSE   = 0.30
    W_FACE   = 0.25
    W_SHARP  = 0.15

    for i, s in enumerate(scores):
        s.motion_score = motions[i]
        s.pose_score = poses[i]
        s.face_score = faces[i]
        s.sharpness_score = sharps[i]
        s.composite_score = (
            W_MOTION * motions[i] +
            W_POSE   * poses[i] +
            W_FACE   * faces[i] +
            W_SHARP  * sharps[i]
        )

    return scores


def get_top_frames(scores: List[FrameScore], n: int = 3) -> List[FrameScore]:
    """
    Return top N frames ensuring temporal diversity
    (no two picks within 1 second of each other).
    """
    sorted_scores = sorted(scores, key=lambda s: s.composite_score, reverse=True)

    selected = []
    for candidate in sorted_scores:
        # Check temporal distance from already selected
        too_close = any(
            abs(candidate.timestamp - s.timestamp) < 1.0
            for s in selected
        )
        if not too_close:
            selected.append(candidate)
        if len(selected) == n:
            break

    # If we couldn't get n diverse frames, just take top n
    if len(selected) < n:
        seen = {s.frame_idx for s in selected}
        for candidate in sorted_scores:
            if candidate.frame_idx not in seen:
                selected.append(candidate)
                seen.add(candidate.frame_idx)
            if len(selected) == n:
                break

    return selected


# ─── ENHANCEMENT (ART component) ───────────────────────────────────────────

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Post-worthy enhancement pipeline:
    1. Intelligent rule-of-thirds crop (subject-aware)
    2. Exposure correction (CLAHE)
    3. Vibrance / saturation boost
    4. Sharpness enhancement
    5. Cinematic color grade (slight warm shadows, cool highlights)
    6. Vignette
    """
    enhanced = frame.copy()

    # 1. Smart crop: detect subject center of mass via pose/saliency
    enhanced = smart_crop(enhanced)

    # 2. CLAHE on luminance channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Vibrance boost (selective saturation — boost less-saturated pixels more)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    vibrance_boost = 1.0 + 0.4 * (1.0 - sat)  # boost unsaturated more
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * vibrance_boost, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Unsharp mask for sharpness
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # 5. Cinematic color grade
    enhanced = cinematic_grade(enhanced)

    # 6. Vignette
    enhanced = apply_vignette(enhanced, strength=0.35)

    return enhanced


def smart_crop(frame: np.ndarray, target_ratio: float = 4/5) -> np.ndarray:
    """
    Crop to portrait/square ratio, centering on detected subject.
    Uses MediaPipe Tasks PoseLandmarker (IMAGE mode) for subject localization.
    """
    h, w = frame.shape[:2]
    current_ratio = h / w

    if abs(current_ratio - target_ratio) < 0.1:
        return frame  # Already close enough

    _ensure_models()
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_options)
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_detector.detect(mp_image)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            xs = [l.x for l in lms]
            ys = [l.y for l in lms]
            cx = int(np.mean(xs) * w)
            cy = int(np.mean(ys) * h)
        else:
            cx, cy = w // 2, h // 2
    finally:
        pose_detector.close()

    # Compute crop dimensions
    if target_ratio > current_ratio:
        # Need taller crop → crop width
        new_w = int(h / target_ratio)
        x1 = max(0, min(cx - new_w // 2, w - new_w))
        return frame[:, x1:x1 + new_w]
    else:
        # Need wider crop → crop height
        new_h = int(w * target_ratio)
        y1 = max(0, min(cy - new_h // 2, h - new_h))
        return frame[y1:y1 + new_h, :]


def cinematic_grade(frame: np.ndarray) -> np.ndarray:
    """
    Apply a cinematic color grade:
    - Warm shadows (lift shadows toward orange-ish)
    - Cool highlights (push highlights toward cyan/teal)
    - Slight contrast S-curve
    """
    img = frame.astype(np.float32) / 255.0

    # S-curve contrast
    img = np.where(img < 0.5,
                   0.5 * (2 * img) ** 1.4,
                   1 - 0.5 * (2 * (1 - img)) ** 1.4)

    # Per-pixel luminance masks (2D) so they broadcast cleanly onto single channels
    lum = img.mean(axis=2)                          # shape (h, w)
    shadow_mask    = np.clip(1.0 - lum * 3, 0, 1)  # bright=0, dark=1
    highlight_mask = np.clip((lum - 0.7) * 3, 0, 1)  # dark=0, bright=1

    # Shadow warmth: in dark areas, push R up slightly, B down slightly
    img[:, :, 2] += shadow_mask * 0.04   # R channel (BGR)
    img[:, :, 1] += shadow_mask * 0.01   # G channel
    img[:, :, 0] -= shadow_mask * 0.02   # B channel

    # Highlight cool: in bright areas, push B up, R down slightly
    img[:, :, 0] += highlight_mask * 0.03  # B
    img[:, :, 2] -= highlight_mask * 0.02  # R

    return np.clip(img * 255, 0, 255).astype(np.uint8)


def apply_vignette(frame: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Apply a smooth radial vignette."""
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = 1 - strength * np.clip(dist - 0.5, 0, 1) * 2

    result = frame.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= vignette

    return np.clip(result, 0, 255).astype(np.uint8)


def frames_to_motion_data(scores: List[FrameScore]):
    """Return timestamps and motion scores for plotting."""
    timestamps = [s.timestamp for s in scores]
    motions = [s.motion_score for s in scores]
    composites = [s.composite_score for s in scores]
    return timestamps, motions, composites
