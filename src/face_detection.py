
import cv2
import numpy as np
import os
import time
from openvino.inference_engine import IECore, IENetwork



class FaceDetectionModel:
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
        self.model_weights = self.model_name.split('.')[0]+'.bin'

    def load_model(self):   
        # initializing IECore object
        self.plugin=IECore() 
        # initializing the network with structure(.xml) and weights(.bin) files
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)

        # extracting useful information from the network for later use
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

        start_time = time.time()
        # creating an executable network, or the IE    
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        end_time = time.time()

        print('FaceDetectionModel Load Time: {}'.format(end_time-start_time))

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


    def predict(self, image, prob_threshold):
        # preprocess the input image(frame in case of video/camera feed)
        processed_image=self.preprocess_input(image.copy())

        infer_start_time = time.time()
        # make inference
        outputs = self.exec_net.infer({self.input_name:processed_image})
        infer_end_time = time.time()

        print('FaceDetectionModel Inference Time: {}'.format(infer_end_time-infer_start_time))

        # getting the coordinates for the cropped face
        coords = self.preprocess_output(outputs, prob_threshold)

        # error checking
        if (len(coords)==0):
            return 0, 0

        coords = coords[0] 
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        # returning the cropped face and its coordinates after preprocessing the outputs
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords

    def preprocess_input(self, image):
        # transforming the input image(frame) so that it is suitable for the model 
        self.image=cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        self.image=self.image.transpose((2, 0, 1))  
        self.image=self.image.reshape(1, *self.image.shape)
       
        return self.image

    def preprocess_output(self, outputs, prob_threshold):
        points =[]
        # generating the coordinates for the cropped face, keeping in mind the probability threshold.

        outs = outputs[self.output_names][0][0]
        for out in outs:
            conf = out[2]
            # if confidence of the model is lower than the specified(or default) probability threshold, then do nothing
            if conf>prob_threshold:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                points.append([x_min,y_min,x_max,y_max])
        return points
