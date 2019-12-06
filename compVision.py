#####################################################################

# dense stereo computer vision for object detection and ranging
# Author: Erdal Guclu

### Code and comments associated with dense stereo vision and camera calibration heavily adapted from article below ###
### For more details check yolo.py ###
"""
@Article{mroz12stereo,
  author = 	 {Mroz, F. and Breckon, T.P.},
  title = 	 {An Empirical Comparison of Real-time Dense Stereo Approaches for use in the Automotive Environment},
  journal =  {EURASIP Journal on Image and Video Processing},
  year =     {2012},
  volume = 	 {2012},
  number = 	 {13},
  pages = 	 {1-19},
  publisher = {Springer},
  url = 	 {http://community.dur.ac.uk/toby.breckon/publications/papers/mroz12stereo.pdf},
  doi = 	 {10.1186/1687-5281-2012-13}
}
"""

### Code related to 3D projection and video file handling adapted from repository below ###
### For more details check stereo_py_3d.py and stereo_disparity.py ###
"""
project SGBM disparity to 3D points for am example pair
of rectified stereo images from a  directory structure
of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

Author : Toby Breckon, toby.breckon@durham.ac.uk
url : https://github.com/tobybreckon/stereo-disparity
Copyright (c) 2017 Deparment of Computer Science,
                   Durham University, UK
License : LGPL - http://www.gnu.org/licenses/lgpl.html
"""


#####################################################################

import cv2
import argparse
import sys
import math
import os
import numpy as np
import glob

# camera calibration data

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

# data location

master_path_to_dataset = "TTBB-durham-02-10-17-sub10"; # ** might want to edit this **
directory_to_cycle_left = "left-images";     # can be edited
directory_to_cycle_right = "right-images";   # can be edited
directory_to_cycle_saved = "saved-images";   # additional directory to put saved images in; can be edited

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);
full_path_directory_saved =  os.path.join(master_path_to_dataset, directory_to_cycle_saved);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

################################################################################

# parse command line arguments for camera ID or video file, and YOLO files
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')
parser.add_argument("-ni", "--nw_input_img", type=int, help="network input image size", default=416)

args = parser.parse_args()

#####################################################################

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = args.nw_input_img       # Width of network's input image
inpHeight = args.nw_input_img      # Height of network's input image

#####################################################################

# dummy on trackbar callback function
def on_trackbar(val):
    return

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, distance, left, top, right, bottom, colour):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2fm' % (class_name, distance)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    centers = {}
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            if classes[classId] == 'car' or classes[classId] == 'truck' or classes[classId] == 'bus' or classes[classId] == 'person':
                confidence = scores[classId]
                if confidence > threshold_confidence:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))
                    centers[(left, top, width, height)] = (center_x, center_y)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []
    centers_nms = {} # the center coordinates of the corresponding bounding box

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])
        centers_nms[boxes[i]] = centers[boxes[i]]

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms, centers_nms)

####################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

####################################################################
    
def makeVideo():
    img_array = []
    for filename in glob.glob(full_path_directory_saved + '.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
     
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

# point_to_3d_dist : project a given disparity point and compute distance
# (uncropped, unscaled) to a set of 3D points with optional colour

def point_to_3d_dist(disparity, max_disparity, x, y):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];
    
    cloudSize = 0
    distance = 0
    
    # compute disparity for a 20x20 cloud
    
    for j in range(-10,11,1):
        for i in range(-10,11,1):
            # if we have a valid non-zero disparity
            try:
                if (disparity[y+j,x+i] > 0):
                    cloudSize += 1
                    
                    # calculate corresponding 3D point [X, Y, Z]
            
                    Z = (f * B) / disparity[y+j,x+i];
                
                    X = ((x - image_centre_w) * Z) / f;
                    Y = ((y - image_centre_h) * Z) / f;
                
                    # compute distance as vector magnitude
                    distance += math.sqrt((X*X)+(Y*Y)+(Z*Z))
            except IndexError:
                # we have entered the region of disparity where the two cameras do not overlap
                pass
    
    if (cloudSize != 0): # there was valid disparity in the cloud
        distance = distance / cloudSize
        
    return distance;

#####################################################################

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

#####################################################################

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

####################################################################

# define video capture object

try:
    # to use a non-buffered camera stream (via a separate thread)

    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture() # not needed for video files

except:
    # if not then just use OpenCV default

    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()

####################################################################

# Load names of classes from file

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

#####################################################################

# define display window name + trackbar

windowNameL = 'left image ' + args.weights_file
windowNameR = 'right image ' + args.weights_file
cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameR, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowNameL , 0, 100, on_trackbar)
#cv2.createTrackbar(trackbarName, windowNameR , 0, 100, on_trackbar)

#####################################################################

# record average computation time per frame
inference_Avg = 0
frame_count = 0

for filename_left in left_file_list:
    # start a timer (to see how long processing and display takes)
    start_t = cv2.getTickCount()
    
    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();
    
    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :
        
        # init the windows for the two image frames
        cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(windowNameR, cv2.WINDOW_NORMAL)
        
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR);
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR);
        
        print("-- files loaded successfully");
        print();

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - perform Contrast Limited Adaptive Histogram Equalization 
        # then raise to the power, as this subjectively appears to improve 
        # subsequent disparity calculation
        
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
        grayL = clahe.apply(grayL)
        grayR = clahe.apply(grayR)
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL,grayR);
        
        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)
        
        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));
        
        ######################################################
        
        # create a 4D tensor (OpenCV 'blob') from the image (pixels scaled 0->1, image resized)
        
        tensorL = cv2.dnn.blobFromImage(imgL, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensorL)
        # runs forward inference to get output of the final output layers
        resultsL = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName,windowNameL) / 100
        classIDs_L, confidences_L, boxes_L, centers_L = postprocess(imgL, resultsL, confThreshold, nmsThreshold)
        
        distances_L = []
        closest_L = ('', 0)
        
        # draw resulting detections on left image
        for detected_object in range(0, len(boxes_L)):
            box = boxes_L[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            # find midpoint of box and compute the distance from car
            midpoint = centers_L[box]
            distance_to_obj = point_to_3d_dist(disparity_scaled, max_disparity, midpoint[0], midpoint[1])
            
            detect_class = classes[classIDs_L[detected_object]] # name of class that was detected
            
            if distance_to_obj != 0: # 0 means that there is no disparity data to compute distance with
                if len(distances_L) > 0:
                    if distance_to_obj <= min(distances_L):
                        closest_L = (detect_class, distance_to_obj)
                else:
                    closest_L = (detect_class, distance_to_obj)
                distances_L.append(distance_to_obj)
            
            # colour the box dependent of the class of the object
            if detect_class == 'car':
                colour = (255, 0, 0)
            elif detect_class == 'truck':
                colour = (0, 255, 255)
            elif detect_class == 'bus':
                colour = (0, 0, 255)
            elif detect_class == 'person':
                colour = (255, 255, 255)
            else:
                colour = (255, 178, 50)
            
            drawPred(imgL, detect_class, distance_to_obj, left, top, left + width, top + height, colour)
            
        # print closest object in each camera to console
        print(filename_left)
        if not closest_L[0] == "":
            print(filename_right + " : nearest detected scene object %s at %.2f meters" % (closest_L[0],closest_L[1]))
        else:
            print(filename_right + " : nearest detected scene object at 0 meters")
        print()
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        inference_Avg += (t * 1000.0 / cv2.getTickFrequency())
        frame_count += 1
        cv2.putText(imgL, label, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.putText(imgR, label, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        ######################################################
        
        cv2.imshow(windowNameL,imgL)
        cv2.imshow(windowNameR,imgR)
        
        # stop the timer and convert to ms. (to see how long processing and display takes for the image pair)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000
        

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space
        if(frame_count == 30):
            break
        
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows
cv2.destroyAllWindows()

print("Average time per frame = %.2fms" % (inference_Avg / frame_count))

#####################################################################


