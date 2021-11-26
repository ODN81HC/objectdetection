import time
import cv2
from base_camera import BaseCamera
import os
import sys

'''
    modules:
        dummy_AI
        YoloOD
'''

class VideoCamera(BaseCamera):
    
    input_source = None
    output = None
    __selected_module__ = ''
    safety = None
    isTiny = None
    labels = []
    
    def __init__(self, input_source, output, modulename, labels, safetyAssist, isTiny):
        VideoCamera.__selected_module__= '{}'.format(modulename)
        VideoCamera.safety = safetyAssist
        VideoCamera.isTiny = isTiny
        VideoCamera.labels = labels
        VideoCamera.input_source = input_source
        VideoCamera.output = output
        super().__init__()
        
    @staticmethod
    def importSmartAssistModule():
        sys.path.append(os.getcwd())
        import importlib.util
        spec = importlib.util.spec_from_file_location(VideoCamera.__selected_module__, '{}/modules/{}.py'.format(os.getcwd(), VideoCamera.__selected_module__))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if VideoCamera.__selected_module__ == 'yoloOD':
            return module.SmartAssistModule(VideoCamera.labels, VideoCamera.isTiny)
        return module.SmartAssistModule()
    

    @staticmethod
    def frames():
        ai_frame = VideoCamera.importSmartAssistModule()

        if VideoCamera.__selected_module__ == 'yoloOD' and VideoCamera.output is not None:
            frameCount = 0
            maxFrameCount = 30
            outVidDir = VideoCamera.output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            outframe = cv2.VideoWriter(outVidDir, fourcc, 20, (640, 480))

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        n_frame = 0
        start_time = time.time()
        # Using OpenCV to capture from either from the camera view, or
        # a camera view
        try:
            video = cv2.VideoCapture(int(VideoCamera.input_source))
        except:
            video = cv2.VideoCapture(VideoCamera.input_source)

        if not video.isOpened():
            raise RuntimeError("Could not start the camera")
        
        while True:
            success, image = video.read()
            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
            if not success:
                break
            else: # Check the producing frames rate of the camera
                n_frame += 1
                elapsed_time = time.time() - start_time
                fps = n_frame/elapsed_time
                cv2.putText(image, "FPS: " + str(round(fps, 2)), (10, 40), font, 1, (255, 255, 255), 2)

            # Control the producing frame rate for process implementation
            if VideoCamera.__selected_module__ == 'dummy_AI':
                frame = ai_frame.img_detect(image)
            else:
                if VideoCamera.safety and VideoCamera.input_source.isdigit():
                    frame, write = ai_frame.img_detect(image, VideoCamera.safety, winLength=11)
                else:
                    frame, write = ai_frame.img_detect(image, VideoCamera.safety, winLength=21)
                if VideoCamera.output is not None:
                    if write:
                        frameCount = 0
                    else:
                        frameCount += 1

                    if frameCount < maxFrameCount:
                        outframe.write(frame)

            _, jpeg = cv2.imencode('.jpg', frame)
            yield jpeg.tobytes()
        
        video.release()