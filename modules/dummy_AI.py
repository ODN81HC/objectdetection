import cv2
import numpy as np

class SmartAssistModule(object):

    def __init__(self):
        self.frame = 0
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.frame_size = (640, 480)

    #Should be implemented in subclass to do whatever processing needs to be done.
    #Default is to just return the frame...
    def img_detect(self, frame):
        height, width, _ = self.load_image(frame)
        if height > self.frame_size[1] and width > self.frame_size[0]:
            self.frame = cv2.resize(self.frame, self.frame_size)
        return self.frame
    
    #Takes the frame from the thread and the implementation can keep copy to process
    def load_image(self, frame):
        self.frame = frame
        height, width, channels = self.frame.shape
        return height, width, channels