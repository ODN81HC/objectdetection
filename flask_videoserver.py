#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python flask_videoserver.py".
# 3. Navigate the browser to the local webpage.
# 4. use http://127.0.0.1:5010/video_feed to watch the video

from flask import Flask, render_template, Response
from camera import VideoCamera
import time
import argparse

#Can be any module from modules directory. Runtime can specify from command line args
default_active_module = 'dummy_AI'
active_module = default_active_module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    target_frames = 120
    n_frames = 0
    prev = 0
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # My implementation that checks the fps of the output
        n_frames += 1
        if n_frames == target_frames:
            fps = target_frames/(time.time() - prev)
            print("The fps of the video streamming: {:.2f}".format(fps))
            prev = time.time()
            n_frames = 0

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(input_source, output, active_module, labels, isSafetyTurnedOn, isTiny)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #Creating an argumnet parser so that we can select the module from the modules directory.
    #By default the base module is used. But then just select the one you want. For examle
    # python .\flask_videoserver.py
    # python .\flask_videoserver.py -m yoloOD
    # python .\flask_videoserver.py -m yoloOD --classes Person,Truck,Car --safetyAssist
    # python .\flask_videoserver.py -m yoloOD --input 0 --output ./videos/result.mp4 --safetyAssist --isTiny
    # python .\flask_videoserver.py -h
    parser = argparse.ArgumentParser(description='Can specify the SmartAssist module to be used with the video source, \
                                                     and select the video source to be used.')
    parser.add_argument('--input', default='./videos/Gantry4.mp4', help='Input source')
    parser.add_argument('--output', default=None, help='Specify the output path if you want to save the stream video')
    parser.add_argument('-m', '--module', default=default_active_module, help='SmartAssist module to use when processing \
                                                                                video. Available modules: dummy_AI, yoloOD')
    parser.add_argument('--classes', default='person', help='[Only work if -m yoloOD] Store class names here')
    parser.add_argument('--isTiny', help='[Only work if -m yoloOD] yolov3 or yolov3-tiny', action='store_true')
    parser.add_argument('--safetyAssist', help='[Only work if -m yoloOD] Turn on Safety Assist \
                                                to get the object movement status', action='store_true')
    args = parser.parse_args()
    
    #Set up which module we will use...
    input_source = args.input
    output = args.output
    active_module = args.module
    isSafetyTurnedOn = args.safetyAssist
    isTiny = args.isTiny
    labels = [i.strip().lower() for i in args.classes.split(',')]
    
    # app.run(host='0.0.0.0', debug=True)
    # app.run(host='127.0.0.1', port=5010, debug=True)
    app.run(host='0.0.0.0', port=5010, debug=True)