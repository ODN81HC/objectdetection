from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
from modules.tracking_algorithm.centroid_track import Track

class Tracker:
    """
    This is the multi-target tracker.
    Parameters:
    -----------

    Attributes:
    -----------
    max_disappeared: int
        The maximum number of frames that a tracker can last
        after missing. After that, it will turn into `Deleted`.
    n_init: int
        The number of frames that turns a `Tentative` state track
        into `Confirm` state. Moreover, if a track disappears
        within this n_init frames, it will turn into `Deleted`.
    """
    def __init__(self, max_disappeared=30, n_init=3):
        self.max_disappeared = max_disappeared
        self.n_init = n_init
        self.tracks = OrderedDict()
        self.next_id = 0

    def register(self, detection, winLength):
        bbox = detection.get_bbox()
        class_name = detection.get_class()
        area = detection.get_area()
        self.tracks[self.next_id] = Track(self.next_id, self.n_init, self.max_disappeared,
                                            bbox, area, winLength, class_name)
        self.next_id += 1
    
    def deregister(self):
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].is_deleted():
                del self.tracks[track_id]
    
    def update(self, detections, winLength):
        """Perform measurement updates and track management.
        Parameters:
        ----------
        detections: List[Detection]
            A list of detections at the current time step
        """
        # If there's no detection found in this frame
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].mark_missed()
        else:
            """If we are currently not tracking any object, take
            the input centroid and register each of them.
            Usually in the inital phase.
            """
            if len(self.tracks) == 0:
                for detection in detections:
                    self.register(detection, winLength)
            else:
                """Otherwise, we are currently tracking objects so we need
                to try to match m-detections into n-trackers by centroid
                distancing approach.
                """
                # Init an array of input centroid for the current frame
                inputcentroids = np.zeros((len(detections), 2), dtype='int')
                # Loop over the bounding box rectangles
                for (i, detection) in enumerate(detections):
                    (x, y, w, h) = detection.get_bbox()
                    cX = int(x + w/2.0)
                    cY = int(y + h/2.0)
                    inputcentroids[i] = (cX, cY)
                # Grab the set of track IDs and corresponding centroids
                track_ids = list(self.tracks.keys())
                trackcentroids = []
                for track in list(self.tracks.values()):
                    # Get the centroids of tracks
                    (x, y, w, h) = track.bbox
                    cX = int(x + w/2.0)
                    cY = int(y + h/2.0)
                    trackcentroids.append((cX, cY))
                D = dist.cdist(np.array(trackcentroids), inputcentroids)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()
                # Loop over the combination of the (row, col) index tuples
                for (row, col) in zip(rows, cols):
                    # If we have already examined either the row/col value
                    # Ignore its val
                    if row in usedRows or col in usedCols:
                        continue
                    # Otherwise, grab the track ID and update accordingly
                    track_id = track_ids[row]
                    self.tracks[track_id].update(detections[col])

                    usedRows.add(row)
                    usedCols.add(col)
                
                # Compute both row and column indices we have NOT examined
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                # if the #obj.centroid >= #input.centroid, check and see
                # if some of these objects have potentially disappeard
                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        # Get track ID
                        track_id = track_ids[row]
                        self.tracks[track_id].mark_missed()
                else:
                    for col in unusedCols:
                        self.register(detections[col], winLength)
        # Deregister tracks
        self.deregister()
