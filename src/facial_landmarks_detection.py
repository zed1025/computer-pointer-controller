
import cv2
import numpy as np
import os
import time
from openvino.inference_engine import IECore, IENetwork



class FacialLandmarksDetectionModel:
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
        # enter the path+filename of the .xml file for the model
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'

    def load_model(self):
        # initializing an IECore object
        self.plugin = IECore()
        # creating a IENetowrk with the model structuer and model weights
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights) 

        # extracting useful information from the network for later use
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

        start_time = time.time()
        # creating an executable network(IE) for inference
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        end_time = time.time()

        print('FacialLandmarksDetectionModel Load Time: {}'.format(end_time-start_time))
        self.check_model()

    def check_model(self):
        # check if all the layers of the model are supported
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        # if all layers of the model are not supported then try to use cpu extension if provided in the command line argument, else exit
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("Error could not be resolved even after adding cpu_extension")
                    exit(1)
            else:
                exit(1)

    def predict(self, image):
        # preprocess the input image(frame in case of video/camera feed)
        preprocess_image = self.preprocess_input(image.copy())

        infer_start_time = time.time()
        # make inference
        outputs = self.exec_net.infer({self.input_name:preprocess_image})
        infer_end_time = time.time()

        print('FacialLandmarksDetectionModel Inference Time: {}'.format(infer_end_time-infer_start_time))

        # getting the coordinates for the output
        # here the output contains among other things the location(coordinates) of the left and the right eye
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        le_xmin=coords[0]-10
        le_ymin=coords[1]-10
        le_xmax=coords[0]+10
        le_ymax=coords[1]+10
        
        re_xmin=coords[2]-10
        re_ymin=coords[3]-10
        re_xmax=coords[2]+10
        re_ymax=coords[3]+10

        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]

        return left_eye, right_eye, eye_coords

    def preprocess_input(self, image):
        # since this model uses a different color space we have to convert it first
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resizing and transposing the image(frame), to use with the model
        image_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))

        # returning the preprocessed input image
        return img_processed

    def preprocess_output(self, outputs):
        # getting the coordinated of the left and right eye from the output
        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)
