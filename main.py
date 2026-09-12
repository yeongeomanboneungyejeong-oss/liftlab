import os
import uuid
import json
import math
import threading
import traceback
import gc
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# Basic paths / app
# ============================================================

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

STATIC = BASE / "static"

app = FastAPI(title="LiftLab AI Biomechanics")

app.mount(
    "/static",
    StaticFiles(directory=STATIC),
    name="static"
)


# ============================================================
# Job / model state
# ============================================================

jobs = {}

jobs_lock = threading.Lock()

model = None
model_lock = threading.Lock()


# ============================================================
# COCO-17 keypoint indices
# ============================================================

K = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,

    "left_shoulder": 5,
    "right_shoulder": 6,

    "left_elbow": 7,
    "right_elbow": 8,

    "left_wrist": 9,
    "right_wrist": 10,

    "left_hip": 11,
    "right_hip": 12,

    "left_knee": 13,
    "right_knee": 14,

    "left_ankle": 15,
    "right_ankle": 16,
}


# ============================================================
# Body segments
# ============================================================

SEGMENTS = {
    "left_upper_arm": (
        "left_shoulder",
        "left_elbow"
    ),

    "left_forearm": (
        "left_elbow",
        "left_wrist"
    ),

    "right_upper_arm": (
        "right_shoulder",
        "right_elbow"
    ),

    "right_forearm": (
        "right_elbow",
        "right_wrist"
    ),

    "left_thigh": (
        "left_hip",
        "left_knee"
    ),

    "left_shin": (
        "left_knee",
        "left_ankle"
    ),

    "right_thigh": (
        "right_hip",
        "right_knee"
    ),

    "right_shin": (
        "right_knee",
        "right_ankle"
    ),

    "torso": (
        "mid_shoulder",
        "mid_hip"
    ),
}


# ============================================================
# Joint angle definitions
# ============================================================

ANGLE_DEFS = {
    "left_elbow": (
        "left_shoulder",
        "left_elbow",
        "left_wrist"
    ),

    "right_elbow": (
        "right_shoulder",
        "right_elbow",
        "right_wrist"
    ),

    "left_shoulder": (
        "left_elbow",
        "left_shoulder",
        "left_hip"
    ),

    "right_shoulder": (
        "right_elbow",
        "right_shoulder",
        "right_hip"
    ),

    "left_hip": (
        "left_shoulder",
        "left_hip",
        "left_knee"
    ),

    "right_hip": (
        "right_shoulder",
        "right_hip",
        "right_knee"
    ),

    "left_knee": (
        "left_hip",
        "left_knee",
        "left_ankle"
    ),

    "right_knee": (
        "right_hip",
        "right_knee",
        "right_ankle"
    ),
}


# ============================================================
# Math helpers
# ============================================================

def angle(a, b, c):
    """
    Calculate angle ABC in degrees.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)

    if na < 1e-6 or nc < 1e-6:
        return None

    cosine = np.dot(ba, bc) / (na * nc)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return round(
        float(
            np.degrees(
                np.arccos(cosine)
            )
        ),
        1
    )


def point_valid(p):
    return (
        p is not None
        and len(p) == 2
        and all(np.isfinite(p))
    )


# ============================================================
# Job helpers
# ============================================================

def set_job(job_id, **values):
    """
    Thread-safe job update.
    """

    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {}

        jobs[job_id].update(values)


def get_job(job_id):
    with jobs_lock:
        return dict(jobs[job_id])


# ============================================================
# YOLO model
# ============================================================

def get_model():

    global model

    if YOLO is None:
        raise RuntimeError(
            "ultralytics가 설치되어 있지 않습니다."
        )

    with model_lock:

        if model is None:

            print(
                "[LiftLab] Loading YOLO pose model...",
                flush=True
            )

            model = YOLO(
                "yolo11n-pose.pt"
            )

            print(
                "[LiftLab] YOLO pose model loaded.",
                flush=True
            )

    return model


# ============================================================
# Frame resizing
# ============================================================

def resize_for_analysis(frame, max_dimension=1280):

    """
    Resize large video frames before inference.

    Returns:
        resized_frame,
        scale_x,
        scale_y
    """

    height, width = frame.shape[:2]

    largest = max(
        width,
        height
    )

    if largest <= max_dimension:

        return (
            frame,
            1.0,
            1.0
        )

    scale = (
        max_dimension / float(largest)
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    scale_x = width / float(new_width)
    scale_y = height / float(new_height)

    return (
        resized,
        scale_x,
        scale_y
    )


# ============================================================
# Coordinate conversion
# ============================================================

def restore_point(
    x,
    y,
    scale_x,
    scale_y
):

    return [
        round(
            float(x) * scale_x,
            2
        ),
        round(
            float(y) * scale_y,
            2
        )
    ]


# ============================================================
# Barbell detection
# ============================================================

def detect_barbell(
    frame,
    joints,
    previous_bar
):

    """
    Heuristic barbell tracker.

    Primary:
        saturated green / cyan / blue plates

    Fallback:
        midpoint between wrists

    Final fallback:
        previous barbell position
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    masks = []

    ranges = [

        # green / cyan
        (
            (35, 70, 60),
            (95, 255, 255)
        ),

        # blue
        (
            (95, 70, 60),
            (135, 255, 255)
        ),
    ]

    for lo, hi in ranges:

        masks.append(
            cv2.inRange(
                hsv,
                np.array(lo),
                np.array(hi)
            )
        )

    mask = cv2.bitwise_or(
        masks[0],
        masks[1]
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        aspect = (
            w / max(h, 1)
        )

        if (
            area > 1200
            and 0.25 < aspect < 4.0
        ):

            candidates.append(
                (
                    area,
                    x + w / 2,
                    y + h / 2
                )
            )

    if candidates:

        _, x, y = max(
            candidates,
            key=lambda item: item[0]
        )

        return [
            round(float(x), 2),
            round(float(y), 2)
        ]

    # Wrist fallback
    if (
        "left_wrist" in joints
        and "right_wrist" in joints
    ):

        return [
            round(
                (
                    joints["left_wrist"][0]
                    +
                    joints["right_wrist"][0]
                ) / 2,
                2
            ),

            round(
                (
                    joints["left_wrist"][1]
                    +
                    joints["right_wrist"][1]
                ) / 2,
                2
            )
        ]

    # Previous frame fallback
    if previous_bar is not None:
        return previous_bar

    return None


# ============================================================
# Main video analysis
# ============================================================

def analyze_video(
    job_id,
    video_path
):

    cap = None

    try:

        print(
            f"[LiftLab] Starting analysis: {job_id}",
            flush=True
        )

        set_job(
            job_id,
            status="analyzing",
            progress=0,
            message="AI pose 분석 준비 중"
        )

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                "영상 파일을 열 수 없습니다."
            )

        fps = (
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 30
        )

        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or 0
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or 0
        )

        print(
            f"[LiftLab] Video: "
            f"{width}x{height}, "
            f"{fps:.2f} FPS, "
            f"{total} frames",
            flush=True
        )

        if total <= 0:

            raise RuntimeError(
                "영상의 프레임 정보를 읽을 수 없습니다."
            )

        # ----------------------------------------------------
        # Load YOLO lazily
        # ----------------------------------------------------

        mdl = get_model()

        frames = []

        previous_bar = None

        # ----------------------------------------------------
        # Process frames
        # ----------------------------------------------------

        for i in range(total):

            ok, frame = cap.read()

            if not ok:
                break

            # -----------------------------------------------
            # Reduce very large input frames
            # -----------------------------------------------

            analysis_frame, scale_x, scale_y = (
                resize_for_analysis(
                    frame,
                    max_dimension=1280
                )
            )

            # -----------------------------------------------
            # YOLO inference
            # -----------------------------------------------

            if torch is not None:

                with torch.inference_mode():

                    result = mdl.predict(
                        analysis_frame,
                        verbose=False,
                        conf=0.25,
                        imgsz=640,
                        device="cpu"
                    )[0]

            else:

                result = mdl.predict(
                    analysis_frame,
                    verbose=False,
                    conf=0.25,
                    imgsz=640,
                    device="cpu"
                )[0]

            joints = {}

            # -----------------------------------------------
            # Pose keypoints
            # -----------------------------------------------

            if (
                result.keypoints is not None
                and len(result.keypoints.xy) > 0
            ):

                xy = (
                    result.keypoints.xy
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------
                # Select largest person
                # -------------------------------------------

                if result.boxes is not None:

                    boxes = (
                        result.boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    areas = (
                        boxes[:, 2]
                        - boxes[:, 0]
                    ) * (
                        boxes[:, 3]
                        - boxes[:, 1]
                    )

                    person_idx = int(
                        np.argmax(areas)
                    )

                else:

                    person_idx = 0

                pts = xy[person_idx]

                # -------------------------------------------
                # Keypoint confidence
                # -------------------------------------------

                if (
                    result.keypoints.conf
                    is not None
                ):

                    confs = (
                        result.keypoints.conf
                        .cpu()
                        .numpy()[person_idx]
                    )

                else:

                    confs = np.ones(
                        17
                    )

                # -------------------------------------------
                # Convert coordinates back to original
                # video resolution
                # -------------------------------------------

                for name, idx in K.items():

                    if idx >= len(pts):
                        continue

                    if confs[idx] < 0.20:
                        continue

                    x = pts[idx][0]
                    y = pts[idx][1]

                    joints[name] = restore_point(
                        x,
                        y,
                        scale_x,
                        scale_y
                    )

            # ------------------------------------------------
            # Mid shoulder
            # ------------------------------------------------

            if (
                "left_shoulder" in joints
                and "right_shoulder" in joints
            ):

                joints["mid_shoulder"] = [

                    round(
                        (
                            joints["left_shoulder"][0]
                            +
                            joints["right_shoulder"][0]
                        ) / 2,
                        2
                    ),

                    round(
                        (
                            joints["left_shoulder"][1]
                            +
                            joints["right_shoulder"][1]
                        ) / 2,
                        2
                    )
                ]

            # ------------------------------------------------
            # Mid hip
            # ------------------------------------------------

            if (
                "left_hip" in joints
                and "right_hip" in joints
            ):

                joints["mid_hip"] = [

                    round(
                        (
                            joints["left_hip"][0]
                            +
                            joints["right_hip"][0]
                        ) / 2,
                        2
                    ),

                    round(
                        (
                            joints["left_hip"][1]
                            +
                            joints["right_hip"][1]
                        ) / 2,
                        2
                    )
                ]

            # ------------------------------------------------
            # Joint angles
            # ------------------------------------------------

            angles = {}

            for name, definition in ANGLE_DEFS.items():

                a, b, c = definition

                if (
                    a in joints
                    and b in joints
                    and c in joints
                ):

                    value = angle(
                        joints[a],
                        joints[b],
                        joints[c]
                    )

                    if value is not None:

                        angles[name] = value

            # ------------------------------------------------
            # Body segments
            # ------------------------------------------------

            segments = {}

            for name, definition in SEGMENTS.items():

                a, b = definition

                if (
                    a in joints
                    and b in joints
                ):

                    mid_x = (
                        joints[a][0]
                        +
                        joints[b][0]
                    ) / 2

                    mid_y = (
                        joints[a][1]
                        +
                        joints[b][1]
                    ) / 2

                    segments[name] = {

                        "a": joints[a],

                        "b": joints[b],

                        "mid": [
                            round(
                                mid_x,
                                2
                            ),
                            round(
                                mid_y,
                                2
                            )
                        ]
                    }

            # ------------------------------------------------
            # Barbell
            # ------------------------------------------------

            bar = detect_barbell(
                analysis_frame,
                joints,
                previous_bar
            )

            # IMPORTANT:
            # barbell was detected in resized coordinates.
            # Convert it back to original video coordinates.

            if bar is not None:

                bar = [
                    round(
                        bar[0] * scale_x,
                        2
                    ),

                    round(
                        bar[1] * scale_y,
                        2
                    )
                ]

                previous_bar = bar

            # ------------------------------------------------
            # Store frame data
            # ------------------------------------------------

            frames.append({

                "frame": i,

                "time": round(
                    i / fps,
                    4
                ),

                "joints": joints,

                "angles": angles,

                "segments": segments,

                "barbell": bar
            })

            # ------------------------------------------------
            # Progress update
            # ------------------------------------------------

            if (
                i % 3 == 0
                or i == total - 1
            ):

                progress = round(
                    (
                        (i + 1)
                        /
                        max(total, 1)
                    ) * 100,
                    1
                )

                message = (
                    f"프레임 분석 중 "
                    f"{i + 1}/{total}"
                )

                set_job(
                    job_id,
                    status="analyzing",
                    progress=progress,
                    message=message
                )

                print(
                    f"[LiftLab] "
                    f"frame {i + 1}/{total} "
                    f"({progress}%)",
                    flush=True
                )

            # ------------------------------------------------
            # Periodic garbage collection
            # ------------------------------------------------

            if i % 30 == 0:

                gc.collect()

        # ----------------------------------------------------
        # Release capture
        # ----------------------------------------------------

        cap.release()
        cap = None

        # ----------------------------------------------------
        # Build final result
        # ----------------------------------------------------

        output = {

            "fps": fps,

            "frame_count": len(frames),

            "width": width,

            "height": height,

            "duration": round(
                len(frames) / fps,
                3
            ),

            "frames": frames
        }

        result_path = (
            DATA
            /
            f"{job_id}.json"
        )

        result_path.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":"
                )
            ),
            encoding="utf-8"
        )

        # ----------------------------------------------------
        # Done
        # ----------------------------------------------------

        set_job(
            job_id,
            status="done",
            progress=100,
            message="분석 완료",
            result=str(
                result_path
            )
        )

        print(
            f"[LiftLab] Analysis complete: {job_id}",
            flush=True
        )

    except Exception as e:

        error_message = str(e)

        print(
            f"[LiftLab] ERROR "
            f"{job_id}: {error_message}",
            flush=True
        )

        traceback.print_exc()

        set_job(
            job_id,
            status="error",
            progress=0,
            message=(
                f"분석 오류: "
                f"{error_message}"
            )
        )

    finally:

        # ----------------------------------------------------
        # Always release OpenCV
        # ----------------------------------------------------

        if cap is not None:

            try:
                cap.release()
            except Exception:
                pass

        # ----------------------------------------------------
        # Cleanup temporary uploaded video
        # ----------------------------------------------------

        try:

            if Path(video_path).exists():

                Path(
                    video_path
                ).unlink()

        except Exception as cleanup_error:

            print(
                "[LiftLab] "
                f"Temporary file cleanup failed: "
                f"{cleanup_error}",
                flush=True
            )

        gc.collect()


# ============================================================
# Front page
# ============================================================

@app.get("/")
def index():

    return FileResponse(
        BASE / "static" / "index.html"
    )


# ============================================================
# Upload / analyze endpoint
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="파일명이 없습니다."
        )

    # --------------------------------------------------------
    # Create job FIRST
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex

    safe_filename = Path(
        file.filename
    ).name

    video_path = (
        DATA
        /
        f"{job_id}_{safe_filename}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Set queued state BEFORE starting thread.
    #
    # This fixes the race condition that could cause
    # "분석 대기 중" to overwrite "analyzing".
    # --------------------------------------------------------

    set_job(
        job_id,
        status="queued",
        progress=0,
        message="분석 대기 중"
    )

    # --------------------------------------------------------
    # Save upload in chunks.
    #
    # Do NOT use:
    #
    #     await file.read()
    #
    # because that loads the whole video into RAM.
    # --------------------------------------------------------

    try:

        total_written = 0

        with open(
            video_path,
            "wb"
        ) as f:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                f.write(chunk)

                total_written += len(
                    chunk
                )

        print(
            f"[LiftLab] Upload complete: "
            f"{safe_filename} "
            f"({total_written / 1024 / 1024:.1f} MB)",
            flush=True
        )

    except Exception as e:

        try:

            if video_path.exists():
                video_path.unlink()

        except Exception:
            pass

        set_job(
            job_id,
            status="error",
            progress=0,
            message=(
                f"업로드 저장 오류: {e}"
            )
        )

        raise HTTPException(
            status_code=500,
            detail=f"영상 저장 실패: {e}"
        )

    finally:

        try:
            await file.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Start background analysis AFTER job exists.
    # --------------------------------------------------------

    thread = threading.Thread(
        target=analyze_video,
        args=(
            job_id,
            video_path
        ),
        daemon=True
    )

    thread.start()

    print(
        f"[LiftLab] "
        f"Analysis thread started: "
        f"{job_id}",
        flush=True
    )

    return {
        "job_id": job_id
    }


# ============================================================
# Job status
# ============================================================

@app.get("/api/status/{job_id}")
def status(
    job_id: str
):

    with jobs_lock:

        if job_id not in jobs:

            raise HTTPException(
                status_code=404,
                detail="작업을 찾을 수 없습니다."
            )

        return dict(
            jobs[job_id]
        )


# ============================================================
# Analysis result
# ============================================================

@app.get("/api/result/{job_id}")
def result(
    job_id: str
):

    result_path = (
        DATA
        /
        f"{job_id}.json"
    )

    if not result_path.exists():

        raise HTTPException(
            status_code=404,
            detail="분석 결과가 아직 없습니다."
        )

    try:

        return json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"결과 읽기 실패: {e}"
               )
