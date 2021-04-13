# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:44:25 2021

@author: tamiryuv
"""
from FaceNet import FaceID
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask import request
import cv2
import imutils
import base64
from PIL import Image
import io
import numpy
from engineio.payload import Payload
import random

td = FaceID()
Payload.max_decode_packets = 50

app = Flask(__name__, static_url_path="", template_folder="./")
app.config['SECRET_KEY'] = 'taesu'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def index():
    print("index.html")
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    # decode and convert into image
    b = io.BytesIO(data_image)
    pil_image = Image.open(b).convert('RGB')
    open_cv_image = numpy.array(pil_image)

    # Convert RGB to BGR
    faces_embeddings = td.embedFace(open_cv_image)  
#    pil_image.save('last.jpg') # if you wanna save image 
    
    emit('response',faces_embeddings)

@app.route('/healthz')
def health():
  return "healthy", 200


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0',port = 8080)