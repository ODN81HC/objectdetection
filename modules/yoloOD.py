import cv2
import numpy as np

from modules.dummy_AI import SmartAssistModule as BaseModule
from modules.tracking_algorithm.centroid_tracker import Tracker
from modules.tracking_algorithm.centroid_detection import Detection

class SmartAssistModule(BaseModule):
    """
    A smart assist module which uses object detection method to detect objects (person, truck,...)
    in the video or a live webcam.
    Moreover, the smart assist also supports the safety assist, which tells the object's movement
    status. The safety assist when turned on, is doing as followed:
    `Green` bounding box indicates that the object is moving away from the camera view.
    `Red` bounding box indicates that the object is moving closer/forward to the camera view.
    `Yellow` bounding box indicates that the object is staying as respected to the camera view.
    """
    def __init__(self, labels, isTiny):
        super().__init__()
        self.labels = labels
        if isTiny:
            yolo_weight = "./yolo-coco/yolov3-tiny.weights"
            yolo_cfg = "./yolo-coco/yolov3-tiny.cfg"
        else:
            yolo_weight = "./yolo-coco/yolov3.weights"
            yolo_cfg = "./yolo-coco/yolov3.cfg"
        self.net = cv2.dnn.readNet(yolo_weight, yolo_cfg)
        self.classes = []
        with open("./yolo-coco/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            layer_names = self.net.getLayerNames()
        if not self.labels:
            self.labels = self.classes
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.confidence_threshold = 0.5
        # Init a modules for this
        self.tracker = Tracker()

    def load_image(self, frame):
        self.frame = frame
        height, width, channels = self.frame.shape
        return height, width, channels

    def detect_object(self):
        blob = cv2.dnn.blobFromImage(self.frame, scalefactor=1/255.0, size=(416, 416),
                                    mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return outs

    def get_boxes_dimension(self, outs, height, width):
        detections = []
        boxes, areas, class_ids, confidences = [], [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    areas.append(detection[2] * detection[3])
        # Apple non-maximum suppresion to get good bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.3)
        # Append necessary attributes into Detection class
        for i in range(len(boxes)):
            if i in indices:
                class_name = str(self.classes[class_ids[i]])
                if class_name in self.labels:
                    detections.append(Detection(boxes[i], areas[i], confidences[i], class_name))
        return detections

    def draw_labels(self, detections, safety, winLength):
        if safety:
            self.tracker.update(detections, winLength)
            for track in list(self.tracker.tracks.values()):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                (x, y, w, h) = track.bbox
                class_name = track.get_class()
                color = track.color
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.frame, class_name, (x, y-5), self.font, 1, color, 2)
        else:
            for detection in detections:
                (x, y, w, h) = detection.get_bbox()
                confidence = detection.get_confidence()
                color = self.colors[self.classes.index(detection.get_class())]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.frame, detection.get_class() + ": " + str(round(confidence, 2)), (x, y - 5),
                            self.font, 1, (255, 255, 255), 2)

    def img_detect(self, frame, safety, winLength):
        height, width, _ = self.load_image(frame)
        outs = self.detect_object()
        detections = self.get_boxes_dimension(outs, height, width)
        if len(detections) == 0:
            write = False
        else:
            write = True
        self.draw_labels(detections, safety, winLength)
        if height > self.frame_size[1] and width > self.frame_size[0]:
            self.frame = cv2.resize(self.frame, self.frame_size)
        return self.frame, write
