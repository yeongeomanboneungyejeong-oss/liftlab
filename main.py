
import os, uuid, json, math, threading, subprocess
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

BASE = Path(__file__).parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="LiftLab Tracking Test")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

jobs = {}
model = None
model_lock = threading.Lock()

# COCO-17 indices
K = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

SEGMENTS = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_thigh": ("right_hip", "right_knee"),
    "right_shin": ("right_knee", "right_ankle"),
    "torso": ("mid_shoulder", "mid_hip"),
}

ANGLE_DEFS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}

def angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a-b, c-b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    x = np.clip(np.dot(ba, bc)/(na*nc), -1, 1)
    return round(float(np.degrees(np.arccos(x))), 1)

def point_valid(p):
    return p is not None and len(p) == 2 and all(np.isfinite(p))

def get_model():
    global model
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    with model_lock:
        if model is None:
            model = YOLO("yolo11n-pose.pt")
    return model

def analyze_video(job_id, video_path):
    try:
        jobs[job_id] = {"status": "analyzing", "progress": 0, "message": "AI pose 분석 준비 중"}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("영상 파일을 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        mdl = get_model()
        frames = []
        previous_bar = None

        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            result = mdl.predict(frame, verbose=False, conf=0.25, imgsz=640)[0]
            joints = {}

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                # Largest detected person
                xy = result.keypoints.xy.cpu().numpy()
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                    person_idx = int(np.argmax(areas))
                else:
                    person_idx = 0
                pts = xy[person_idx]
                confs = result.keypoints.conf.cpu().numpy()[person_idx] if result.keypoints.conf is not None else np.ones(17)
                for name, idx in K.items():
                    if idx < len(pts) and confs[idx] >= 0.20:
                        joints[name] = [round(float(pts[idx][0]), 2), round(float(pts[idx][1]), 2)]

            if "left_shoulder" in joints and "right_shoulder" in joints:
                joints["mid_shoulder"] = [
                    round((joints["left_shoulder"][0]+joints["right_shoulder"][0])/2,2),
                    round((joints["left_shoulder"][1]+joints["right_shoulder"][1])/2,2)
                ]
            if "left_hip" in joints and "right_hip" in joints:
                joints["mid_hip"] = [
                    round((joints["left_hip"][0]+joints["right_hip"][0])/2,2),
                    round((joints["left_hip"][1]+joints["right_hip"][1])/2,2)
                ]

            angles = {}
            for name, (a,b,c) in ANGLE_DEFS.items():
                if all(x in joints for x in (a,b,c)):
                    v = angle(joints[a], joints[b], joints[c])
                    if v is not None:
                        angles[name] = v

            # Segment midpoint + endpoints. This is the data used for segment trails.
            segments = {}
            for name, (a,b) in SEGMENTS.items():
                if a in joints and b in joints:
                    segments[name] = {
                        "a": joints[a], "b": joints[b],
                        "mid": [
                            round((joints[a][0]+joints[b][0])/2,2),
                            round((joints[a][1]+joints[b][1])/2,2)
                        ]
                    }

            # Heuristic barbell tracking:
            # detect large saturated green/blue regions; otherwise use wrist midpoint.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks = []
            for lo, hi in [
                ((35,70,60),(95,255,255)),  # green/cyan/blue-green
                ((95,70,60),(135,255,255)), # blue
            ]:
                masks.append(cv2.inRange(hsv, np.array(lo), np.array(hi)))
            mask = cv2.bitwise_or(masks[0], masks[1])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > 1200 and 0.25 < w/max(h,1) < 4.0:
                    candidates.append((area, x+w/2, y+h/2))
            bar = None
            if candidates:
                # Prefer the largest colored plate region near the lower/middle body.
                _, x, y = max(candidates, key=lambda z:z[0])
                bar = [round(float(x),2), round(float(y),2)]
            elif "left_wrist" in joints and "right_wrist" in joints:
                bar = [
                    round((joints["left_wrist"][0]+joints["right_wrist"][0])/2,2),
                    round((joints["left_wrist"][1]+joints["right_wrist"][1])/2,2)
                ]
            elif previous_bar is not None:
                bar = previous_bar
            if bar is not None:
                previous_bar = bar

            frames.append({
                "frame": i,
                "time": round(i/fps, 4),
                "joints": joints,
                "angles": angles,
                "segments": segments,
                "barbell": bar
            })

            if i % 3 == 0:
                jobs[job_id]["progress"] = round((i+1)/max(total,1)*100, 1)
                jobs[job_id]["message"] = f"프레임 분석 중 {i+1}/{total}"

        cap.release()
        out = {
            "fps": fps, "frame_count": len(frames),
            "width": width, "height": height,
            "duration": round(len(frames)/fps,3),
            "frames": frames
        }
        result_path = DATA / f"{job_id}.json"
        result_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        jobs[job_id] = {"status":"done","progress":100,"message":"분석 완료","result":str(result_path)}
    except Exception as e:
        jobs[job_id] = {"status":"error","progress":0,"message":str(e)}

@app.get("/")
def index():
    return FileResponse(BASE/"static/index.html")

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "파일명이 없습니다.")
    job_id = uuid.uuid4().hex
    video_path = DATA / f"{job_id}_{Path(file.filename).name}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    threading.Thread(target=analyze_video, args=(job_id, video_path), daemon=True).start()
    jobs[job_id] = {"status":"queued","progress":0,"message":"분석 대기 중"}
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return jobs[job_id]

@app.get("/api/result/{job_id}")
def result(job_id: str):
    p = DATA / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404, "분석 결과가 아직 없습니다.")
    return json.loads(p.read_text(encoding="utf-8"))
