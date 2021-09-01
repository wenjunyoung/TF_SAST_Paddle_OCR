#!/usr/bin/env python
import cv2
import time
import argparse 
import numpy as np
from post_process import resize_image_type2, PostProcess
                       
parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml', type=str, required=True, help='path of the .xml file from model')
parser.add_argument('-b', '--bin', type=str, required=True, help='path of the .bin file from model')
args = vars(parser.parse_args())

model_xml = args["xml"]
model_bin = args["bin"]
net = cv2.dnn.readNet(model_xml, model_bin)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

outNames = net.getUnconnectedOutLayersNames()

while True:

    frame = vs.read()
    shape_list = resize_image_type2(640, frame)
    blob = cv2.dnn.blobFromImage(frame, size=(640,640), mean=(104.04,113.985,119.85), scalefactor=1/255., swapRB=True)
    
    # 前向推理
    net.setInput(blob)
    output = net.forward(outNames)
    # 后处理
    post_result = PostProcess(output, shape_list)
    boxes = post_result[0]['points']

    if len(boxes) > 0:
        for box in boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [box], True, color=(255, 255, 0), thickness=2)

    cv2.imshow("Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        break
    elif key == ord('q'):
        exit()
