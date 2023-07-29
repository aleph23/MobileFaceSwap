import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo

class LandmarkModel():
    '''LandmarkModel: This is the main class of this module. It is used to load the model and prepare it for inference.'''
    
    def __init__(self, name, root='./checkpoints'):
        '''
        The constructor takes two arguments:
            name: The name of the model.
            root: The path to the directory where the model is stored.
        
        model_zoo:  This submodule contains a function called get_model which returns a model object based on the path to its .onnx file.
        model:  This submodule contains a class called Model which is used to load and prepare a model for inference. It has two methods:
            prepare: This method takes one argument, ctx_id, and prepares the model for inference on the device with that context id.
            detect: This method takes four arguments, img, threshold, max_num and metric, and performs inference on the given image using the given threshold and maximum number of faces and returns bounding boxes and landmarks for each face detected in the image.
            Note that this class also has a predict method which is not used in this project. It is used to perform inference on a single face only.
        '''

        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        '''
        The prepare method takes three arguments:
            ctx_id: The context id of the device on which the model will be run.
            det_thresh: The threshold for face detection.
            det_size: The size of the image that will be fed to the face detection model.
        '''
    
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        '''
        The get method takes two arguments:
            img: The image on which inference will be performed.
            max_num: Maximum number of faces that can be detected in an image.
            kps is the landmarks of the face with the highest detection score.

        '''

        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        return kpss[best_index] if kpss is not None else None

    def gets(self, img, max_num=0):
        '''gets method which is not used in this project. It is used to get all the landmarks in an image.'''
    
        bboxes, kpss = self.det_model.detect(img, threshold=self.det_thresh, max_num=max_num, metric='default')
        return kpss
        