# !/usr/bin/python -tt
# Author: Shashwat Pathak

# Demo at: https://eternalreader.pythonanywhere.com/

import os
from app import app
from flask import flash, request, redirect, url_for, render_template,send_from_directory
from werkzeug.utils import secure_filename
from tools import ToolBox
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'tiff']) # This set checks the valid image types supported by
# ImagEd. Although, tiff files are supported for processing, their visibility is only compatible with Safari browser.


def allowed_file(filename): # Ensures that the uploaded file is an image among the allowed types
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if 'file' not in request.files: # Checking if file part is present in the request or not
        flash('No file part') # Flash message if not file part is found
        return redirect(request.url) # Redirect to the same url
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading') # If no file is being uploaded then redirect to the same url
        return redirect(request.url)
    if file and allowed_file(file.filename): # If file exits and the filename is of correct format
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):  # After every new upload, the upload directory
            # is cleared of all the previous enhancements done on the previously loaded image
            for nem in files:
                os.remove(os.path.join(root, nem))
        # We need to create a secure filename, before storing the image in order to adhere to the security vulnerabiliti
        # -es if the original file path is rendered. Thus we create an encoded alias to the paths of the external image
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Saving the uploaded Image
        actual_image = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename),format='jpg')
        print('Image Recieved')
        image = cv2.resize(actual_image, (512,512)) # resizing the uploaded image using OpenCV
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename), image, format='jpg')  # saving the resized image
        # into the upload folder
        print('Image Saved')
        flash('Image successfully uploaded and displayed below')
        # rendering the secured output image location for it to be displayed in the image editor.
        return render_template('home.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


# Function/Endpoint for handling the display of the uploaded image.
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)  # renders the url of the image for dis
    # play


# function for performing histogram equalization
@app.route('/equalize/<filename>', methods=['GET', 'POST'])
def histogram_equalization(filename):
    image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
    # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    path = image_files[0]
    image = plt.imread(path, format='jpg')
    print(np.shape(image))

    # creating the toolbox object and accessing the histogram equalization function.
    equalized = ToolBox(image).histogram_equalization()

    # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
    # the previous image and, the original uploaded image.
    for i in range(1, len(image_files) - 1):
        os.remove(image_files[i])

    eq_filename = 'eq_image_' + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
    # Saving the processed image
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], eq_filename), equalized, format='jpg')

    return eq_filename


# Function for performing log transform and rendering the output to the UI
@app.route('/logtransform/<filename>', methods=['GET', 'POST'])
def log_transform(filename):
    image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
    # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    path = image_files[0]
    image = plt.imread(path, format='jpg')
    print(np.shape(image))

    # creating the toolbox object and accessing the log function.
    log_transformed = ToolBox(image).log_transform()

    # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
    # the previous image and, the original uploaded image.
    for i in range(1, len(image_files) - 1):
        os.remove(image_files[i])

    log_filename = 'log_image_' + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], log_filename), log_transformed, format='jpg')

    return log_filename


# Function for performing sharpening on the image and rendering the output to the UI
@app.route('/sharpening/<filename>', methods=['GET', 'POST'])
def sharpening(filename):
    sharp_filename = ""
    if request.method == 'POST':
        alpha = float(request.form['alpha'])
        beta = float(request.form['beta'])
        image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
        # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        path = image_files[0]
        image = plt.imread(path, format='jpg')
        print(np.shape(image))

        # creating the toolbox object and accessing the sharpening function.
        sharpened = ToolBox(image).sharpening(alpha, beta)

        # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
        # the previous image and, the original uploaded image.
        for i in range(1,len(image_files) - 1):
            os.remove(image_files[i])

        sharp_filename = 'sharp_image_' + str(alpha).split('.')[0] + str(alpha).split('.')[1] + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], sharp_filename), sharpened, format='jpg')

    return sharp_filename


# Function for performing gamma correction on the image and rendering the output to the UI
@app.route('/gamma/<filename>', methods=['GET', 'POST'])
def gamma_correction(filename):
    gamma_filename = ""
    if request.method == 'POST':
        gamma = float(request.form['gamma'])
        print(gamma)
        image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
        # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        path = image_files[0]
        image = plt.imread(path, format='jpg')
        print(np.shape(image))

        # creating the toolbox object and accessing the gamma transformed function.
        gamma_transformed = ToolBox(image).gamma_transformation(gamma)

        # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
        # the previous image and, the original uploaded image.
        for i in range(1,len(image_files) - 1):
            os.remove(image_files[i])

        gamma_filename = 'gamma_image_' + str(gamma).split('.')[0] + str(gamma).split('.')[1] + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'],
                                gamma_filename), gamma_transformed, format='jpg')

    return gamma_filename


# Function for performing blurring on the image and rendering the output to the UI
@app.route('/blur/<filename>', methods=['GET', 'POST'])
def blurring(filename):
    blur_filename = ""
    if request.method == 'POST':
        kernel_size = int(request.form['kernel_size'])
        sigma = float(request.form['sigma'])
        # print(window_size)
        image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
        # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        path = image_files[0]
        image = plt.imread(path, format='jpg')
        print(np.shape(image))

        # creating the toolbox object and accessing the gaussian blurring function.
        gaussian_blurred = ToolBox(image).gaussian_blur(sigma, kernel_size)

        # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
        # the previous image and, the original uploaded image.
        for i in range(1,len(image_files) - 1):
            os.remove(image_files[i])

        blur_filename = 'blur_image' + '_' + str(kernel_size) + '_' + str(sigma).split('.')[0] + str(sigma).split('.')[1] + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], blur_filename), gaussian_blurred, format='jpg')

    return blur_filename


# Additional Feature
@app.route('/median/<filename>', methods=['GET', 'POST'])
def median_filtering(filename):
    median_filename = ""
    if request.method == 'POST':
        window_size = int(request.form['window_size_median'])
        # print(window_size)
        image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
        # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        path = image_files[0]
        image = plt.imread(path, format='jpg')
        print(np.shape(image))

        # creating the toolbox object and accessing the median filtering function.
        median_filtered = ToolBox(image).median_filtering(window_size)

        # Keeping the size of the file stack fixed, since we want to keep only three images in buffer. The current image,
        # the previous image and, the original uploaded image.
        for i in range(1,len(image_files) - 1):
            os.remove(image_files[i])

        median_filename = 'median_image' + '_' + str(window_size) + path.split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], median_filename), median_filtered, format='jpg')
    return median_filename


# Function for rendering the original uploaded image and removing all the past enhancements from the upload directory
# Here we are performing the reset functions, that means we wish to clear all the previous enhancements and only the
# original uploaded image should be left in the stack.
@app.route('/reset/', methods=['GET', 'POST'])
def reset():
    # Accessing filenames of the processed images saved in the uploads folder, sorted in increasing order of their
    # modification time (newest on top)
    image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))

    # removing all the previous image except the original,.
    for i in range(len(image_files) -1):
        os.remove(image_files[i])
    filename = image_files[len(image_files)-1].split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
    return filename


# Function for reverting back to the previous enhancement
# here we need to take care of the special case when we undo, when we have only a single image.
@app.route('/undo/', methods=['GET', 'POST'])
def undo():
    # Accessing filenames of the processed images saved in the uploads folder, sorted in increasing order of their
    # modification time (newest on top)
    image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))

    # if number of images in the stack is greater than 1, then remove the current image and render the previous image
    if len(image_files) > 1:
        os.remove(image_files[0])
        filename = image_files[1].split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
    else:
        filename = image_files[0].split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
        flash('Cannot Undo Anymore, you have reached the root image')

    return filename


# Function for downloading the image
@app.route('/uploads/', methods=['GET', 'POST'])
def download():
    directory = app.config['UPLOAD_FOLDER']
    image_files = list(reversed(sorted(glob.glob(app.config['UPLOAD_FOLDER'] + '*'), key=os.path.getmtime)))
    filename = image_files[0].split(app.config['UPLOAD_FOLDER'][:-1] + '\\')[1]
    # path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(directory=directory, path=filename, as_attachment=True)
    # return path

# function for rendering the profile page
@app.route('/about/')
def about():
    return render_template("about.html")


# function for displaying the output
# This function is active only in the deployed version of ImagEd (GUI for image processing) which can be found at
# https://eternalreader.pythonanywhere.com/ (Link to the Deployed tool)
@app.route('/about/display/')
def profile_display():
    display_image_filename = app.config['PROFILE_PIC']
    return redirect(url_for('static', filename=display_image_filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
