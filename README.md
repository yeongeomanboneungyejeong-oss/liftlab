# LiftLab Tracking Test

실제 영상 파일을 업로드하면 서버에서 YOLO Pose를 실행해:
- 프레임별 관절 좌표
- 관절각
- 신체 분절 endpoint/midpoint
- 바벨 위치(색상 plate heuristic + wrist fallback)

를 JSON으로 생성합니다.

## 실행
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

브라우저에서 http://127.0.0.1:8000

## Docker
```bash
docker build -t liftlab .
docker run -p 7860:7860 liftlab
```

주의: 바벨 검출은 범용 heuristic입니다. 실제 서비스에서는 바벨 전용 detector/tracker로 교체하는 것이 좋습니다.
