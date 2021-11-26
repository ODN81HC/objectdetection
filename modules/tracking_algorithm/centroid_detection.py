class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Attributes
    ----------
    bbox: (x, y, w, h)
        Detector bounding box
    area: float
        Detector bounding box's area
        (normalized by diving to frame_height*frame_width)
    confidence : float
        Detector confidence score.
    class_name : float
        Detector class.
    """

    def __init__(self, bbox, area, confidence, class_name):
        self.bbox = bbox
        self.area = area
        self.confidence = confidence
        self.class_name = class_name

    def get_class(self):
        return self.class_name

    def get_bbox(self):
        return self.bbox

    def get_area(self):
        return self.area
    
    def get_confidence(self):
        return self.confidence
