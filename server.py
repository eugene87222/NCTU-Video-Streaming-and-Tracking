# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import argparse
import cv2 as cv
from flask import Flask, Response, request

from detectors import Detector
from utils import Timer, draw_tracks, select_track
from trackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker

app = Flask(__name__, template_folder='./')

camera = cv.VideoCapture('/dev/video10', cv.CAP_V4L)
camera.set(cv.CAP_PROP_BUFFERSIZE, 2)
# camera.set(cv.CAP_PROP_FPS, 25)

height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
width  = camera.get(cv.CAP_PROP_FRAME_WIDTH)
rescale_size = 800
if height > width:
    width = rescale_size / height * width
    height = 800
else:
    height = rescale_size / width * height
    width = 800

fps = camera.get(cv.CAP_PROP_FPS)

model, tracker = None, None

tracks, trk_id = None, None
# target_cid = [0]
# target_cid = [0, 2]
# target_cid = [1, 2, 3, 5, 7]
target_cid = None

timer = Timer()


def capture():
    global tracks
    while True:
        timer.tic()
        ok, image = camera.read()
        if ok:
            image, bboxes, confidences, class_ids = model.detect(image)
            tracks = tracker.update(bboxes, confidences, class_ids)
            updated_image = draw_tracks(image.copy(), tracks, trk_id, target_cid)

            duration = timer.toc(average=True)

            cv.putText(updated_image, f'Frame: {tracker.frame_count}',
                    (1, 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)
            cv.putText(updated_image, f'FPS (video): {fps:.2f}',
                    (1, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)
            cv.putText(updated_image, f'FPS (tracker): {(1/duration):.2f}',
                    (1, 45), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)

            ret, jpeg = cv.imencode('.jpg', updated_image)
            frame =  jpeg.tobytes()
        else:
            camera.release()
            break
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n'
               b'\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(capture(), mimetype='multipart/x-mixed-replace; boundary=frame')


def rescale(x, y, h, w):
    
    x = int(x) / int(w) * width
    y = int(y) / int(h) * height
    return int(x), int(y)


@app.route('/data')
def data():
    global tracks, trk_id
    coor = request.args.get('coor')
    if coor == 'deselect':
        trk_id = None
        print(f'Deselect')
    else:
        x, y, h, w = coor.split(',')
        x, y = rescale(x, y, h, w)
        trk_id = select_track(x, y, target_cid, tracks)
        print(f'Click @ {x}/{width}, {y}/{height}')
        print(f'target: {trk_id} @ #frame {tracker.frame_count}')
    return json.dumps({'success': True}), 200, {'ContentType':'application/json'} 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object detections in input video using YOLOv3 trained on COCO dataset')

    parser.add_argument('--video', '-v', type=str, help='Input video path')

    parser.add_argument('--model', type=str,
                        default='yolo_weights/yolov3.cfg',
                        help='Path to model definition file.')

    parser.add_argument('--weights', type=str,
                        default='yolo_weights/yolov3.weights',
                        help='Path to weights file.')

    parser.add_argument('--class_names', type=str,
                        default='yolo_weights/coco.names',
                        help='Path to class label file.')

    parser.add_argument('--tracker', '-t', type=str,
                        default='IOUTracker',
                        help='Tracker used to track objects. Options include [\'CentroidTracker\', \'CentroidKF_Tracker\', \'IOUTracker\', \'SORT\']. Default is \'IOUTracker\'')

    parser.add_argument('--conf_thres', type=float,
                        default=0.6,
                        help='Object confidence threshold.')

    parser.add_argument('--nms_thres', type=float,
                        default=0.4,
                        help='IoU thresshold for non-maximum suppression.')

    parser.add_argument('--yolo_input_size', type=int,
                        default=512)

    args = parser.parse_args()
    # print(json.dumps(vars(args), indent=2))

    logging.basicConfig(
        format='%(levelname)s, %(asctime)s, %(funcName)s, %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)

    if args.tracker.lower() == 'centroidtracker':
        tracker = CentroidTracker(
            max_lost=10, tracker_output_format='mot_challenge')
    elif args.tracker.lower() == 'centroidkf_tracker':
        tracker = CentroidKF_Tracker(
            max_lost=10, tracker_output_format='mot_challenge',
            centroid_distance_threshold=50,
            process_noise_scale=0.5,
            measurement_noise_scale=0.5)
    elif args.tracker.lower() == 'sort':
        tracker = SORT(
            max_lost=10, tracker_output_format='mot_challenge',
            iou_threshold=0.2,
            process_noise_scale=0.5,
            measurement_noise_scale=0.5)
    elif args.tracker.lower() == 'ioutracker':
        tracker = IOUTracker(
            max_lost=10, tracker_output_format='mot_challenge',
            iou_threshold=0.2,
            min_detection_confidence=0.4)
    else:
        raise NotImplementedError

    model = Detector(
        model=args.model,
        weights=args.weights,
        class_names=args.class_names,
        yolo_input_size=args.yolo_input_size,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres
    )

    app.run(host='0.0.0.0', debug=True)
