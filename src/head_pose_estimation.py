
import cv2
import numpy as np
import os
from openvino.inference_engine import IECore, IENetwork


class HeadPoseEstimationModel:
    def __init__(self, model_name, device='CPU', extensions=None):
        # class variables
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None

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

        start_time = time.time()
        # creating an executable network(IE) for inference
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        end_time = time.time()

        print('FaceDetectionModel Load Time: {}'.format(end_time-start_time))

        # extracting useful information from the network for later use
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = [i for i in self.network.outputs.keys()]

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
        img_processed = self.preprocess_input(image.copy())
        # make inference
        outputs = self.exec_net.infer({self.input_name:img_processed})
        # preprocessing the outputs - i.e. the result we get after inference
        finalOutput = self.preprocess_output(outputs)
        return finalOutput


    def preprocess_input(self, image):
        # resizing and transposing the image(frame), to use with the model
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed

    def preprocess_output(self, outputs):
        outs = []
        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])
        return outs
