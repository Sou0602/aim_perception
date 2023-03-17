import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def extract_video(path,images_root):

    cap = cv2.VideoCapture(path)

    ret = True
    count = 0
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imsave(images_root + f'{count:04d}'+'.jpg',img)
            count += 1
		
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', nargs='+', type=str, default='ball_tracking_video.mp4',     help='input video path')
    parser.add_argument('--images-root', nargs='+', type=str, default='images/', help='images root directory')

    args = parser.parse_args()

    extract_video(args.input_video,args.images_root[0])



