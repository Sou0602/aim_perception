# aim_perception
Aim Long Form Perception Exercise

## Task 1 
**Usage:**
python task1.py --Grid Grid(str) --Orientations Orientations(str) --FoV FoV(str) 

**Sample Input (These values are also stored as default inputs):**

python task1.py --Grid '[[0,0,0,0,0],[T,0,0,0,2],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0]]' --Orientations
'[180,150]' --FoV '[60,60]'

## Task 2

**To Run the files:**

1.	Include the images folder from the google drive folder shared or run the task2_extract.py file after creating images folder and copying ball_tracking_video.mp4 to task2 folder.
2.	Copy the weights file from weights in google drive folder to the yolov7 directory.
3.	To run track_yolo_kcf.py, run the file from the task2 folder, default parameters can be changed from command line.
4.	To run track_yolo_csrt.py, run the file from the task2 folder, default parameters can be changed from command line.
5.	To run yolov7/detect.py, copy ball_tracking_video.mp4 to yolov7 directory and run detect.py from yolov7 directory. If –save-txt variable is used while running, labels are generated and stored in runs/exps_num/labels/
6.	To run yolo_makecsv.py, I also have labels for one of the experiment runs, exp8 on the repo to run the files. But change the path to generate a different csv. 
7.	If using for a different video, bad_frames need to be changed. I refined the detect.py , specifically to soccer_ball.py (which needs to be changed for other applications. )

