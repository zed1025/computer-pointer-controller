
import cv2
import numpy as np
import os, math
from openvino.inference_engine import IECore, IENetwork


class GazeEstimationModel:
    def __init__(self, model_name, device='CPU', extensions=None):
        # class variables
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'

    def load_model(self):
        # initializing an IECore object
        self.plugin = IECore()
        # creating a IENetowrk with the model structuer and model weights
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        # creating an executable network(IE) for inference
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)

        # extracting useful information from the network for later use
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]


    def predict(self, left_eye_image, right_eye_image, hpa):
        # preprocessing the left eye, and the right eye image
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        # feeding the outputs from headposeestimation, and facedetection to the model for inference
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        # preprocessing the outputs from the model, and getting the location for mouse pointer
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector

    def preprocess_input(self, left_eye, right_eye):
        # we have to preprocess both the lefteye and righteye image
        le_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        re_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_resized,axis=0), (0,3,1,2))
        re_img_processed = np.transpose(np.expand_dims(re_image_resized,axis=0), (0,3,1,2))
        return le_img_processed, re_img_processed

    def preprocess_output(self, outputs, hpa):
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (newx,newy), gaze_vector
