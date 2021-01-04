## Environment

- Ubuntu 18.04 or higher
- Python 3.7.5

## Requirements

- ffmpeg
- `pip install -r requirements.txt`

## Steps
1. Execute `python -m http.server` under project directory
2. Execute `python server.py`
3. Execute `python stream.py`
4. Open `http://localhost:8000/index/index-hls.html` in the browser

## WebSocket
1. Execute `python websocket.py` under project directory
2. Execute `python -m http.server`
3. Open `http://localhost:8000/Click_Socket.html`

## Ref.

- https://video.aminyazdanpanah.com/python/start
- https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
