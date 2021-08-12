# !/usr/bin/python -tt

import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
import numpy as np
from PIL import Image
import scipy.signal as sig
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import math
import requests
from scipy import ndimage
import cv2
import glob

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for nem in files:
                os.remove(os.path.join(root, nem))

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        actual_image = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = cv2.resize(actual_image, (350,350)) # Image for display
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename), image, format='png')
        flash('Image successfully uploaded and displayed below')
        return render_template('home.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/about/')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)