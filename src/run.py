
import cv2
import os, logging
import numpy as np

from argparse import ArgumentParser

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel

from mouse_controller import MouseController
from input_feeder import InputFeeder

'''
Use the command below to run the code in CPU mode. Check README.md for additional command line arguments

python3 run.py -f /home/amit/dev/ov_workspace/computer-pointer-controller-master/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl /home/amit/dev/ov_workspace/computer-pointer-controller-master/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp /home/amit/dev/ov_workspace/computer-pointer-controller-master/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -ge /home/amit/dev/ov_workspace/computer-pointer-controller-master/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i /home/amit/dev/ov_workspace/computer-pointer-controller-master/bin/demo.mp4
'''

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="path to .xml file of Face-Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="path to .xml file of Facial-Landmark-Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="path to .xml file of Head-Pose-Estimation model.")
    parser.add_argument("-ge", "--gazeestimationmodel", required=True, type=str,
                        help="path to .xml file of Gaze-Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path to cpu_extension file. Not needed in v2020.x")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="probability threshold to compare against model confidence")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="specify target decive, CPU is default")
    
    return parser

def main():
    args = build_argparser().parse_args()
    
    logger = logging.getLogger()
    inputFilePath = args.input
    inputFeeder = None
    if inputFilePath.lower()=="cam":
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    
    modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarksDetectionModel':args.faciallandmarkmodel, 
    'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    
    frame_count = 0
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_out = hpem.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break
    logger.error("Video ended, exiting...")
    cv2.destroyAllWindows()
    inputFeeder.close()
     
    

if __name__ == '__main__':
    main() 
 
