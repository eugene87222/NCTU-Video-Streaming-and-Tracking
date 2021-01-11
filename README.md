## Environment

- Ubuntu 18.04 or higher
- Python 3.7.5

## Requirements

- ffmpeg
- PyTorch 1.7+
- `pip install -r requirements.txt`

### Download YOLOv3 pretrained weight

```
mkdir yolo_weights
wget -P yolo_weights https://pjreddie.com/media/files/yolov3.weights
wget -P yolo_weights https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget -P yolo_weights https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -P yolo_weights https://raw.githubusercontent.com/adipandas/multi-object-tracker/master/examples/pretrained_models/yolo_weights/coco_names.json
```

## Steps
1. Go to project directory
2. Execute `python -m http.server`
3. Execute `python server.py`
4. Execute `python websocket.py`
5. Execute `python stream.py`
6. Open `http://localhost:8000/index.html`

To shutdown Flask server, go to `http://localhost:5000/shutdown`

## Reference

- UI design:
  - https://freshman.tech/custom-html5-video/
  - https://www.w3schools.com/css/css_dropdowns.asp
- HLS streaming: https://video.aminyazdanpanah.com/python/start
- Flask server: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
