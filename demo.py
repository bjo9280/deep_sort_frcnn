#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
# from yolo import YOLO
import tensorflow as tf


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

PATH_TO_INFERENCE_GRAPH= 'export_frcnn/frozen_inference_graph.pb'


def inference_frames(sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, image, width, height, threshold):

    image_np = np.array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # bboxes = get_info_from_DF(int(file_name.split(".")[0]), original)
    # Actual detection.
    tic = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    toc = time.time()
    t_diff = toc - tic
    fps = 1 / t_diff


    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)
    return_boxs = []

    for i in range(int(num[0])):
        # if classes[i] != 2.0:
        if scores[i] < threshold:
            continue

        x = int(boxes[i][1] * width)
        y = int(boxes[i][0] * height)
        w = int((boxes[i][3] * width) - (boxes[i][1] * width))
        h = int((boxes[i][2] * height) - (boxes[i][0] * height))
        if x < 0 :
            w = w + x
            x = 0
        if y < 0 :
            h = h + y
            y = 0
        # return_boxs.append([boxes[i][1] * width, boxes[i][0] * height, boxes[i][3] * width, boxes[i][2] * height])
        return_boxs.append([x,y,w,h])

    return return_boxs

def main():

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('input.mp4')

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output2.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    print('Importing graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_INFERENCE_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Use temp list of dictionaries to hold output data
    pre_track_data = []

    # Generate a video object

    print('Starting session...')
    output = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Define input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            fps = 0.0
            while True:
                ret, frame = video_capture.read()  # frame shape 640*480*3
                if ret != True:
                    break
                t1 = time.time()

               # image = Image.fromarray(frame)
                image = Image.fromarray(frame[...,::-1]) #bgr to rgb

                boxs = inference_frames(sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, image, image.width, image.height, 0.8)

                features = encoder(frame,boxs)

                # score to 1.0 here).
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)


                for det in detections:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

                cv2.imshow('', frame)

                if writeVideo_flag:
                    # save a frame
                    out.write(frame)
                    frame_index = frame_index + 1
                    list_file.write(str(frame_index)+' ')
                    if len(boxs) != 0:
                        for i in range(0,len(boxs)):
                            list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
                    list_file.write('\n')

                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %f"%(fps))

                # Press Q to stop!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
