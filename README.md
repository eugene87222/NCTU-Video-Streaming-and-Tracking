## Environment

- Ubuntu 18.04 or higher
- Python 3.7.5

## Requirements

- ffmpeg
- `pip install -r requirements.txt`

## Steps
1. Go to project directory
2. Execute `python -m http.server`
3. Execute `python server.py`
4. Execute `python websocket.py`
5. Execute `python stream.py`
6. Open `http://localhost:8000/index-hls.html`

## WebSocket + Click Position
1. Go to project directory
2. Execute `python websocket.py`
3. Execute `python -m http.server`
4. Open `http://localhost:8000/Click_Socket.html`

## Reference

- UI: https://freshman.tech/custom-html5-video/
- HLS streaming: https://video.aminyazdanpanah.com/python/start
- Flask server: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
