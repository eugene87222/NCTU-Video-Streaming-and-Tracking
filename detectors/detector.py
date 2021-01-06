import torch
import cv2 as cv
import numpy as np

from .model import Darknet
from utils import load_classes, non_max_suppression, rescale_boxes, pad_to_square_tensor, resize


class Detector():
    def __init__(self, model, yolo_input_size, class_names, weights, conf_thres, nms_thres):
        self.yolo_input_size = yolo_input_size
        self.nms_thres = nms_thres  
        self.conf_thres = conf_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes = load_classes(class_names)

        self.model = Darknet(model, img_size=self.yolo_input_size).to(self.device)
        self.model.load_darknet_weights(weights)
        self.model.eval()

    def detect(self, image):
        rescale_size = 800
        h, w = image.shape[0:2]
        if h > w:
            factor = rescale_size / h
        else:
            factor = rescale_size / w
        image = cv.resize(image, None, fx=factor, fy=factor)

        h, w, _ = image.shape
        _image, pad = pad_to_square_tensor(image, 0)
        _image = resize(_image, self.yolo_input_size).unsqueeze(0)

        with torch.no_grad():
            detections = self.model(_image.to(self.device))
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        bboxes = []
        confidences = []
        class_ids = []

        if detections[0] is not None:
            detections = rescale_boxes(detections[0], self.yolo_input_size, [h, w])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if (x2-x1)*(y2-y1) >= 0.65*w*h:
                    continue
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w-1, int(x2))
                y2 = min(h-1, int(y2))
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bboxes.append((x1, y1, bbox_w, bbox_h))
                confidences.append(cls_conf)
                class_ids.append(int(cls_pred))

        return image, np.array(bboxes), np.array(confidences), class_ids
