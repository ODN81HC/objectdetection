from scipy.signal import savgol_filter

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

class TrackColor:
    """
    Enumeration type for the single target track color. New created tracks are
    classified as `Yellow` until it changes to either `Green` or `Red`.
    The `Red` color indicates that the track has been move closer to the camera view
    The `Green` color indicates that the track has been move further to the camera
    view.

    """
    Yellow = (0, 255, 255)
    Green = (0, 128, 0)
    Red = (0, 0, 255)

class Track:
    """
    A single target track with centroid update.
    Parameters:
    -----------
    hits: int
        Number of frames since the track exists
    time_since_update: int
        Number of frame that the track will be `Deleted`
        if time_since_update > max_disappeared


    Attributes:
    -----------
    track_id: int
        Unique track id
    bbox: Tuple(x, y, w, h)
        The coordinates of the track at current frame
    area: float
        The area of the track according to the bbox and the frame
        The formula is: (w*h) / (fheight*fwidth)
    class_name: str
        The class name of the track
    state: TrackState(object)
        The current state of the track
    color: TrackColor(object)
        The current color of the track
    """
    def __init__(self, track_id, n_init, max_disappeared, bbox, area, winLength, class_name=None):
        self.track_id = track_id
        self._n_init = n_init
        self.max_disappeared = max_disappeared
        self.bbox = bbox
        self.winLength = winLength
        self.area = [area] * self.winLength
        self.class_name = class_name

        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.color = TrackColor.Yellow
    
    def get_class(self):
        return self.class_name
    
    def update(self, detection):
        """Perform update on tracker by centroid distancing method.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        
        # Get the new area, new bbox, new class name
        new_area = detection.get_area()
        self.bbox = detection.get_bbox()
        self.class_name = detection.get_class()

        if len(self.area) == self.winLength:
            del self.area[0]
            self.area.append(new_area)
        else:
            self.area = [new_area] * self.winLength
        # Update the color in each track
        self.update_color()
    
    def update_color(self):
        """Perform update the track color to change the status of the bounding box
        """
        filtered_data = savgol_filter(self.area, self.winLength, 1)
        if self.winLength == 21:
            last_state_area = filtered_data[self.winLength//2 - 8]
        else:
            last_state_area = filtered_data[self.winLength//2 - 4]

        # Conditional for color update
        """This conditional update is based on the ratio of the current frame compared to
        the last frame's area.
        The conditional is tested based on multiple tests only with Person class.
        """
        if filtered_data[-1] <= 0.08:
            if filtered_data[-1] / last_state_area <= 0.992:
                self.color = TrackColor.Green
            elif filtered_data[-1] / last_state_area >= 1.05:
                self.color = TrackColor.Red
            else:
                self.color = TrackColor.Yellow
        elif 0.08 < filtered_data[-1] <= 0.2:
            if filtered_data[-1] / last_state_area <= 0.985:
                self.color = TrackColor.Green
            elif filtered_data[-1] / last_state_area >= 1.1:
                self.color = TrackColor.Red
            else:
                self.color = TrackColor.Yellow
        else:
            if filtered_data[-1] / last_state_area <= 0.87:
                self.color = TrackColor.Green
            elif filtered_data[-1] / last_state_area >= 1.19:
                self.color = TrackColor.Red
            else:
                self.color = TrackColor.Yellow
        # Reassign to the track's area list again
        self.area = filtered_data.tolist()
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.time_since_update += 1
        # if self.state == TrackState.Tentative:
        #     self.state = TrackState.Deleted
        if self.time_since_update > self.max_disappeared:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
