import numpy as np
import os
import argparse

def make_csv_from_yolo_preds(input_video,labels_path,len_yolo_video,output_csv,W,H,run_window,second_ball_thres):

    root_dir = os.getcwd() + '/' + labels_path
    files = os.listdir(root_dir)
    video_ext = input_video.split('.')[0] + '_'
    len_video = len_yolo_video
    store_arr = np.zeros((len(files),5))
    p = 0
    bad_frames = []
    bad_frames_ball_count = []
    for k in range(1,len_video+1):
        file_name = video_ext + str(k) + '.txt'
        if file_name not in files:
            continue
        else:
            with open(root_dir+ '/' + file_name) as f:
                lines = f.readlines()
                ball_frame = 0
                xs = []
                ys = []
                ws = []
                hs = []
                for l in lines:
                    if l.split(' ')[0] == '32': ## If its a sports ball
                        frame_id = k-1 ## to start from 0
                        if ball_frame == 0:
                            p += 1
                        ball_frame += 1

                        ## check for more than 2 balls in one frame
                        ## keep a running average of the x,y positions and the size
                        ## eliminate big differences

                        x = float(l.split(' ')[1]) * W ##x_center of the bb box
                        y = float(l.split(' ')[2]) * H ##y_center of the bb box
                        w = float(l.split(' ')[3]) * W ## width
                        h = float(l.split(' ')[4]) * H ## height
                        xs.append(x)
                        ys.append(y)
                        ws.append(w)
                        hs.append(h)

                if ball_frame > 1 and p > 1:
                    ## current update of x,y,w,h

                    sq_dist = (xs - run_x)**2 + (ys - run_y)**2
                    min_ind = np.argmin(sq_dist)
                    x = xs[min_ind]
                    y = ys[min_ind]
                    w = ws[min_ind]
                    h = hs[min_ind]

                store_arr[p-1][0] = frame_id
                store_arr[p-1][1] = x
                store_arr[p-1][2] = y
                store_arr[p-1][3] = w
                store_arr[p-1][4] = h

                if w*h < second_ball_thres: ## check for the second ball
                    if ball_frame > 0:
                        bad_frames.append(frame_id)
                        bad_frames_ball_count.append(ball_frame)
        ## Running Mean of x and y
        if p-1 <= run_window:
            run_x = store_arr[:run_window,1].mean()
            run_y = store_arr[:run_window,2].mean()
        else:
            run_x = store_arr[p-run_window:p,1].mean()
            run_y = store_arr[p - run_window:p,2].mean()

    store_arr = store_arr[np.sum(store_arr,1)>0]
    for f in bad_frames:
        store_ind = np.where(store_arr[:,0] == f)
        np.delete(store_arr,store_ind,axis=0)

    np.savetxt(output_csv,store_arr,delimiter=",")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-video', nargs='+', type=str, default='ball_tracking_video.mp4',help='input video path')
    parser.add_argument('--output-csv', type=str, default='video_frames_yolo.csv', help='output csv file')
    parser.add_argument('--labels-path', type=str, default='yolov7/runs/detect/exp8/labels', help='path to labels')
    parser.add_argument('--W', type=int, default=1280, help='Frame Width')
    parser.add_argument('--H', type=int, default=720,     help='Frame Height')
    parser.add_argument('--len-yolo', type=int, default=958, help='Frames in Yolo video')
    parser.add_argument('--run-window', type=int, default=2,     help='Running Mean Window')
    parser.add_argument('--second-thres', type=int, default=500, help='Sq distance threshold for second ball')
    args = parser.parse_args()

    make_csv_from_yolo_preds(args.input_video,args.labels_path,args.len_yolo,args.output_csv,
                             args.W,args.H,args.run_window,args.second_thres)






