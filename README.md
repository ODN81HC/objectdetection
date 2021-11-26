## SmartAssist (Flask) video streaming application

This is a test framework which allows various different image processing modules to be used from a source camera or video, and apply various techniques to them. 

The directory structure is as follows:
modules - Different modules that can be used to process the video feed.
templates - The HTML templates for the browser
videos - Any videos that we wish to use for processing can be stored here
yolo-coco - The yolo configuration and weights files.

The video streaming application is made so that when turning off the browser, the application will be on paused as well, and when re-opening the browser will start the application again. Here is the [link](https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited)

It is running on a thread so that the video camera frames will be synchronized with the Flask requesting frame, hence save the computer's resources

The smartassist modules will be put inside the folder `modules` and the weights and config files will be put inside the `yolo-coco` folder

## How to use:

(a) Download the weigths and names into the yolo-coco directory:
- Download the weights and config file for yolo object detection here: [weight](https://github.com/smarthomefans/darknet-test/blob/master/yolov3-tiny.weights) [config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
- Or if you want a better object recognition (will be traded off with the real-time), you can download [here](https://pjreddie.com/media/files/yolov3.weights) for the weight and [here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) for the config file.
- Download the coco dataset name [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

- In `modules/yoloOD.py`, it is running using yoloV3-tiny by default, if you want to change it into yoloV3 for a better object detection process, you can change this line (line 11): 
```python
self.net = cv2.dnn.readNet("./yolo-coco/yolov3-tiny.weights", "./yolo-coco/yolov3-tiny.cfg")
```
```python
self.net = cv2.dnn.readNet("./yolo-coco/yolov3.weights", "./yolo-coco/yolov3.cfg")
```
(b) Make sure the right module is selected that you want to run:
- In `camera.py`, line 10 is the init of the class that we want to **replace the modules into**, and line 39 is where the camera input frame is processed to get the output frame. Therefore, remember to replace it with other modules so that it can run properly.

(c) Start the server to capture video or view the video file
- `python flask_videoserver.py` for the default video streamming Flask application
- `python flask_videoserver.py -m yoloOD` YoloV3 object detection on webcam
- `python flask_videoserver.py --input .videos/Gantry4.mp4 -m yoloOD` Yolov3 object detection on video
- `python flask_videoserver.py -m yoloOD --safetyAssist` YoloV3 with object status detection on webcam
- `python flask_videoserver.py -m yoloOD --safetyAssist --isTiny` YoloV3-tiny with object detection on webcam
- `python flask_videoserver.py -m yoloOD --input 0 --output ./videos/result.mp4 --safetyAssist --isTiny`
    export the video into the output with detected objects involved

(d) use one of the following in the browser to view the results
-templates/index.html
-http://127.0.0.1:5010/video_feed
