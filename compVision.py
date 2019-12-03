#####################################################################

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import argparse
import sys
import math
import os
import numpy as np

# camera calibration data

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

# where is the data ? - set this to where you have it

master_path_to_dataset = "TTBB-durham-02-10-17-sub10"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

#SURFACE-DETECTION###################################################

# check if the OpenCV we are using has the extra modules available

def extraOpenCVModulesPresent():

    # we only need to check this once and remember the result
    # so we can do this via a stored function attribute (static variable)
    # which is preserved across calls

    if not hasattr(extraOpenCVModulesPresent, "already_checked"):
        (is_built, not_built) = cv2.getBuildInformation().split("Disabled:")
        extraOpenCVModulesPresent.already_checked = ('xfeatures2d' in is_built)

    return extraOpenCVModulesPresent.already_checked

################################################################################

keep_processing = True

# parse command line arguments for camera ID or video file, and YOLO files
parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true', help="run in full screen mode")
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
parser.add_argument("-cl", "--class_file", type=str, help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str, help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str, help="network weights", default='yolov3.weights')

args = parser.parse_args()

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

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 348       # Width of network's input image
inpHeight = 348      # Height of network's input image

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

#PROJECT-TO-3D#######################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x];

                X = ((x - image_centre_w) * Z) / f;
                Y = ((y - image_centre_h) * Z) / f;

                # add to points

                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);

    return points;

## point_to_3d_dist : project a given disparity point and compute distance
## (uncropped, unscaled) to a set of 3D points with optional colour

def point_to_3d_dist(disparity, max_disparity, x, y):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);

    distance = -1

    # if we have a valid non-zero disparity
    
    if (disparity[y,x] > 0):

        # calculate corresponding 3D point [X, Y, Z]

        # stereo lecture - slide 22 + 25

        Z = (f * B) / disparity[y,x];

        X = ((x - image_centre_w) * Z) / f;
        Y = ((y - image_centre_h) * Z) / f;
        #print(x,y, disparity[y,x])
        
        # compute distance as vector magnitude
        distance = math.sqrt((X*X)+(Y*Y)+(Z*Z))
    return distance;

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    # calc. Zmax as per above

    # Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h;
        points2.append([x,y]);

    return points2;

#####################################################################

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

#full_path_filename_left = os.path.join(full_path_directory_left, "1506942480.483420_L.png");
#full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R");

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

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

#####################################################################

# define display window name + trackbar

windowNameL = 'left image ' + args.weights_file
windowNameR = 'right image ' + args.weights_file
cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameL, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowNameL , 0, 100, on_trackbar)
cv2.createTrackbar(trackbarName, windowNameR , 0, 100, on_trackbar)

#####################################################################

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

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR) #args.video_file = imgL, imgR
        
        # bilateral pass to reduce noise
        imgL = cv2.bilateralFilter(imgL,3,30,1)
        imgR = cv2.bilateralFilter(imgR,3,30,1)
        
        print("-- files loaded successfully");
        print();

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY); #args.video_file = grayL, grayR
        
        #grayL = cv2.Laplacian(grayL, cv2.CV_16S, ksize=3)
        #grayR = cv2.Laplacian(grayL, cv2.CV_16S, ksize=3)

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

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

        # crop disparity to chop out left part where there are with no disparity
        # as this area is not seen by both cameras and also
        # chop out the bottom area (where we see the front of car bonnet)

        if (crop_disparity):
            width = np.size(disparity_scaled, 1);
            disparity_scaled = disparity_scaled[0:390,135:width];

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));
        
        # project to a 3D colour point cloud (with or without colour)
        
        #points = project_disparity_to_3d(disparity_scaled, max_disparity, imgL);
        
        ######################################################
        
        # create a 4D tensor for each image (OpenCV 'blob') from the image (pixels scaled 0->1, image resized)
        tensorL = cv2.dnn.blobFromImage(imgL, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        tensorR = cv2.dnn.blobFromImage(imgR, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensorL)
        # runs forward inference to get output of the final output layers
        resultsL = net.forward(output_layer_names)
        # repeat for right image
        net.setInput(tensorR)
        resultsR = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName,windowNameL) / 100
        classIDs_L, confidences_L, boxes_L, centers_L = postprocess(imgL, resultsL, confThreshold, nmsThreshold)
        classIDs_R, confidences_R, boxes_R, centers_R = postprocess(imgR, resultsR, confThreshold, nmsThreshold)
        
        distances_L = []
        distances_R = []
        closest_L = ('', 0)
        closest_R = ('', 0)
        # draw resulting detections on left image
        for detected_object in range(0, len(boxes_L)):
            box = boxes_L[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            #find midpoint of box and compute the distance from car
            midpoint = centers_L[box]
            distance_to_obj = point_to_3d_dist(disparity_scaled, max_disparity, midpoint[0], midpoint[1])
            
            detect_class = classes[classIDs_L[detected_object]] # name of class that was detected
            
            if distance_to_obj != -1: # -1 means that there is no disparity data to compute distance with4
                if len(distances_L) > 0:
                    if distance_to_obj == max(distances_L):
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
            
        # draw resulting detections on right image
        for detected_object in range(0, len(boxes_R)):
            box = boxes_R[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            detect_class = classes[classIDs_R[detected_object]]
            
            # find midpoint of box and compute the distance from car
            midpoint = centers_R[box]
            distance_to_obj = point_to_3d_dist(disparity_scaled, max_disparity, midpoint[0], midpoint[1])
            
            detect_class = classes[classIDs_R[detected_object]] # name of class that was detected
            
            if distance_to_obj != -1: # -1 means that there is no disparity data to compute distance with
                if(len(distances_R) > 0):
                    if distance_to_obj == max(distances_R):
                        closest_R = (detect_class, distance_to_obj)
                else:
                    closest_R = (detect_class, distance_to_obj)
                distances_R.append(distance_to_obj)
            
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
            
            drawPred(imgR, detect_class, distance_to_obj, left, top, left + width, top + height, colour)
            
        # print closest object in each camera to console
        if not closest_L[0] == "":
            print("Closest object left camera: %s at %.2f meters" % (closest_L[0],closest_L[1]))
        else:
            print("No valid distance estimate to report on left camera")
        if not closest_R[0] == "":
            print("Closest object right camera: %s at %.2f meters" % (closest_R[0],closest_R[1]))
        else:
            print("No valid distance estimate to report on right camera")
        print()
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
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
        # crop - c
        # pause - space
        
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows

cv2.destroyAllWindows()

#####################################################################


