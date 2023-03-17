import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys
import numpy as np
import os
path = os.getcwd()
sys.path.append(path+'/yolov7')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox


def track(input_video_path):
    """Tracks the bounding boxes, given a video file."""

    ##setup yolo
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    tracker = cv2.TrackerKCF_create()
    video = cv2.VideoCapture(input_video_path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    out = cv2.VideoWriter(opt.output_video,
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print( 'Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (889, 542, 89, 89)

    ok = tracker.init(frame, bbox)
    start = time.time()
    frame_id = 0
    store_arr = []
    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        if not ok or bbox[0] < 0 or bbox[1] < 0:

            bbox = query_yolo(opt.images_root+str(frame_id).zfill(4)+'.jpg',
                              imgsz,stride,device,half,model,frame_id)

            if bbox == -1:

                frame_id += 1
                out.write(frame)
                continue
            else:
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame, bbox)

        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            store_arr.append([frame_id,int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2),bbox[2],bbox[3]])
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        frame_id += 1
        out.write(frame)
        if frame_id == len(os.listdir(opt.images_root)):
            break
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    video.release()
    print(time.time() - start)
    # Close all frames and video windows.
    cv2.destroyAllWindows()
    np.savetxt(opt.output_csv,np.array(store_arr),delimiter=",")

def query_yolo(source,imgsz,stride,device,half,model,frame_id):
    """ Returns the yolo detections if found in a image"""

    im0s = cv2.imread(source)
    img = letterbox(im0s, imgsz, stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


    # Process detections
    ball_detections = []
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                    ## ignore bad frames
                    if int(cls) == 32 and frame_id not in [630, 631, 659, 742, 751, 753, 782, 833]:
                         ball_detections.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])))

    if len(ball_detections) == 0:
        return -1

    else:
        return ball_detections[0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', nargs='+', type=str, default='ball_tracking_video.mp4', help='input video path')
    parser.add_argument('--images-root', nargs='+', type=str, default='images/', help='images root')
    parser.add_argument('--output-video', nargs='+', type=str, default='ball_tracking_kcf_yolo.mp4', help='output video path')
    parser.add_argument('--output-csv', nargs='+', type=str, default='video_frames_kcf.csv',help='output csv')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='yolov7/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7/yolov7.pt']:
                track()
                strip_optimizer(opt.weights)
        else:
            track(opt.input_video)
