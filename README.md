# Computer-Vision

Command line argument options in order of parsing:
-c : specify camera to use (default=0) type=int
video_file : specify optional video file (optional metavar) type=str
-cl : file holding list of classes (default=coco.names) type=str
-cf : network config file (default=yolo3.cfg) type=str
-w : network weights file (default=yolo3.weights) type=str
-ni : network input image size (default=416) type=int

Note for option -ni only the values 128,256,416,832 have been tested, other values might result
in errors.

In order to run the script with default settings simply navigate to the directory and type
the command compVision.py on windows (./compVision.py). In order to pass parameters use the
options described above followed by what every parameter. e.g "compVision.py -ni 256" will
set the network input size to 256x256.

To supply video file by default the program will search for directory "TTBB-durham-02-10-17-sub10"
within the same directory as the source and will look for the subdirectories "left-images" and
"right-images". So specify a custom path to the video frames modify 
master_path_to_dataset = "TTBB-durham-02-10-17-sub10" on line 59.

During runtime there are several options:
-press 'x' to exit at anytime
-press 's' to save the disparity, left and right image 
 (will create sgbm-disparity.png, left.png and right.png in source directory)
-press 'space' to pause playback on next frame

Credits to relevant authors can be found in the comments at the top of compVision.py