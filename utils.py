import numpy as np
from lsnms import nms
import pytesseract
import cv2
import re

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def preprocess_yolo(image, input_shape=(640,640)):
    img_height, img_width = image.shape[:2]

    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img = cv2.resize(input_img, input_shape)

    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    # input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    input_tensor = input_img.astype(np.float32)

    

    return input_tensor

def postprocess_yolo(output, 
                conf_threshold,
                iou_threshold, 
                orig_shape):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = extract_boxes(predictions, orig_shape)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    keep = nms(boxes, scores, iou_threshold=iou_threshold, class_ids=class_ids)
            
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    number_boxes = boxes[class_ids == 0]
    sum_boxes = boxes[class_ids == 1]
    sign_boxes = boxes[class_ids == 2]

    number_scores = scores[class_ids == 0]
    sum_scores = scores[class_ids == 1]
    sign_scores = scores[class_ids == 2]

    number_box = number_boxes[np.argmax(number_scores)] if len (number_scores) else []
    sum_box = sum_boxes[np.argmax(sum_scores)] if len (sum_scores) else []
    sign_box = sign_boxes[np.argmax(sign_scores)] if len (sign_scores) else []


    return number_box, sum_box, sign_box


def extract_boxes(predictions, orig_shape, exp=5):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, orig_shape)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    expanded_boxes = []
    for box in boxes:
        x1,y1,x2,y2 = box
        expanded_boxes.append([x1-exp, y1-exp, x2+exp, y2+exp])

    return np.array(expanded_boxes)

def rescale_boxes(boxes,
                  orig_shape,
                  input_shape=(640,640),
                  ):

    # Rescale boxes to original image dimensions
    input_height, input_width = input_shape
    img_height, img_width = orig_shape

    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes


def ocr(orig_bgr_image,
        num_box,
        sum_box,
        num_alphabet = 'AБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        sum_alphabet = '0123456789.,'):
    num_box = [int(var) for var in num_box]
    sum_box = [int(var) for var in sum_box]

    img_rgb = cv2.cvtColor(orig_bgr_image, cv2.COLOR_BGR2RGB)
    num_crop = img_rgb[num_box[1]:num_box[3], num_box[0]:num_box[2], :]
    sum_crop = img_rgb[sum_box[1]:sum_box[3], sum_box[0]:sum_box[2], :]


    # options = "-l {}".format('rus+eng')
    options = "-l {}".format('eng')

    num_string = pytesseract.image_to_string(num_crop, config=options)
    sum_string = pytesseract.image_to_string(sum_crop, config=options)

    num_string = postprocess_tes(num_string)
    sum_string = postprocess_tes(sum_string)

    num_string_pattern = re.findall(r'\D{1,5}\d{3,10}', num_string)
    num_string =  num_string_pattern[0] if len(num_string_pattern) else num_string

    num_string = ''.join([char for char in num_string if char in num_alphabet])
    sum_string = ''.join([char for char in sum_string if char in sum_alphabet])

    return num_string, sum_string

def postprocess_tes(string, alphabet='AБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.'):
    string = string.replace('\n', '')
    string = string.replace('\x0c', '')
    for char in string:
        if not (char in alphabet):
            string = string.replace(char, '')
    return string