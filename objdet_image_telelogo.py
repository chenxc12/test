# -*- coding: utf-8 -*-

import cv2
import numpy as np
from onnxruntime import InferenceSession
from utils1 import load_classes, preprocess, postprocess, plot_results


if __name__ == "__main__":
    
    # get input data
    input_size = 640
    img_file = "/Users/sssssssssss/Downloads/yolov5-master/data/telelogo_test/03.jpg"
    img = cv2.imread(img_file)
    # img_h = img.shape[0]
    # img_w = img.shape[1]
    input_data, offset = preprocess(img, input_size, input_size)
    # print(input_data)
    # print(input_data.shape)
    # print(offset)

    # get classes
    class_name_file = "/Users/sssssssssss/Downloads/yolov5-master/data/telelogo_test/coco.names"
    classes = load_classes(class_name_file)
    # classes = ["telelogo"]

    # inference

    output_names = ["output0"]
    onnx_file = "/Users/sssssssssss/Downloads/yolov5-master/data/telelogo_test/telecom-logo-detect-simp.onnx"
    session = InferenceSession(onnx_file, providers='CPUExecutionProvider')
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names=output_names, input_feed={"images": input_data})
    # print(outputs)
    # print(outputs[0].shape)

    # postprocess
    pred_boxes, pred_cls, pred_scores = postprocess(outputs,
                                                    offset, 
                                                    conf_thresh=0.25, 
                                                    iou_thresh=0.45)
    # print(pred_boxes)
    # print(pred_cls)
    # print(pred_scores)

    # show result
    save_path = "./output/pred_result_onnx.jpg"
    plot_results(img, pred_boxes, pred_cls, pred_scores, class_names=classes, color=None, save_path=save_path)
