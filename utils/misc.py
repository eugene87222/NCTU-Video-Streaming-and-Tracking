import cv2 as cv
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    with open(path, 'r') as fp:
        names = fp.read().split('\n')[:-1]
    return names


def rescale_boxes(boxes, current_dim, original_shape):
    ''' Rescales bounding boxes to the original shape '''
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def get_centroid(bboxes):
    '''
    Calculate centroids for multiple bounding boxes.

    Parameters
    ----------
    bboxes : numpy.ndarray
        Array of shape `(n, 4)` or of shape `(4,)`.
        Where each row contains `(xmin, ymin, width, height)`.

    Returns
    -------
    numpy.ndarray : Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    '''

    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    w, h = bboxes[:, 2], bboxes[:, 3]

    xc = xmin + 0.5*w
    yc = ymin + 0.5*h

    x = np.hstack([xc[:, None], yc[:, None]])
    if one_bbox:
        x = x.flatten()
    return x


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1*h1+1e-16) + w2*h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2]/2, box1[:, 0] + box1[:, 2]/2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3]/2, box1[:, 1] + box1[:, 3]/2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2]/2, box2[:, 0] + box2[:, 2]/2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3]/2, box2[:, 1] + box2[:, 3]/2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def iou(bbox1, bbox2):
    '''
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Parameters
    ----------
    bbox1 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, x-bottom-right, y-bottom-right) of length 4.
    bbox2 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, x-bottom-right, y-bottom-right) of length 4.

    Returns
    -------
    iou: float
         intersection-over-onion of bbox1, bbox2.
    '''

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    if overlap_x1-overlap_x0<=0 or overlap_y1-overlap_y0<=0:
        return 0.0

    size_1 = (x1_1-x0_1) * (y1_1-y0_1)
    size_2 = (x1_2-x0_2) * (y1_2-y0_2)
    size_intersection = (overlap_x1-overlap_x0) * (overlap_y1-overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    iou_ = size_intersection / size_union
    return iou_


def iou_xywh(bbox1, bbox2):
    '''
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Parameters
    ----------
    bbox1 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, width, height) of length 4.
    bbox2 : numpy.array, list of floats
            bounding box in format (x-top-left, y-top-left, width, height) of length 4.

    Returns
    -------
    iou: float
         intersection-over-onion of bbox1, bbox2.
    '''
    bbox1 = bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]
    bbox2 = bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]
    iou_ = iou(bbox1, bbox2)
    return iou_


def xywh2xyxy(xywh):
    xyxy = xywh.new(xywh.shape)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    '''
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.

    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    '''
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def get_color(idx):
    idx = idx * 3
    color = ((37*idx)%255, (17*idx)%255, (29*idx)%255)
    return color


def draw_tracks(image, tracks, trk_id=None, target_cid=None):
    '''
    Draw on input image.

    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.

    Returns:
        numpy.ndarray : image with the track-ids drawn on it.
    '''

    if trk_id is None:
        for trk in tracks:
            trk_id = trk[1]
            bb = trk[2:6]
            conf = trk[6]
            cid = trk[7]

            if target_cid is not None and cid not in target_cid:
                continue

            clr = get_color(trk_id)
            trk_id = str(trk_id)
            cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            (label_width, label_height), baseLine = cv.getTextSize(trk_id, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv.rectangle(image,
                         (bb[0], y_label-label_height),
                         (bb[0]+label_width, y_label+baseLine),
                         (255, 255, 255), cv.FILLED)
            cv.putText(image, trk_id, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
    else:
        for trk in tracks:
            if trk[1] == trk_id:
                bb = trk[2:6]
                conf = trk[6]
                cid = trk[7]

                if target_cid is not None and cid not in target_cid:
                    break

                clr = get_color(trk_id)
                trk_id = str(trk_id)
                cv.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
                (label_width, label_height), baseLine = cv.getTextSize(trk_id, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_label = max(bb[1], label_height)
                cv.rectangle(image,
                             (bb[0], y_label - label_height),
                             (bb[0] + label_width, y_label + baseLine),
                             (255, 255, 255), cv.FILLED)
                cv.putText(image, trk_id, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
                break
    return image


def select_track(x, y, target_cid, tracks):
    for trk in tracks:
        xmin, ymin, w, h = trk[2:6]
        if (target_cid is None or trk[7] in target_cid) and x>=xmin and x<=xmin+w and y>=ymin and y<=ymin+h:
            return trk[1]
    return None


def pad_to_square_tensor(image, pad_value):
    image = transforms.ToTensor()(image)
    c, h, w = image.shape
    dim_diff = np.abs(h-w)
    pad1, pad2 = dim_diff//2, dim_diff - dim_diff//2
    pad = (0, 0, pad1, pad2) if h<=w else (pad1, pad2, 0, 0)
    image = F.pad(image, pad, 'constant', value=pad_value)
    return image, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode='nearest').squeeze(0)
    return image
