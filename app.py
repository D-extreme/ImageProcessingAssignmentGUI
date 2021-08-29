# Author: Shashwat Pathak

from flask import Flask

# Folder for saving the enhancements
UPLOAD_FOLDER = 'C:/Users/shash/PycharmProjects/ImageProcessingAssignmentGUI/static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROFILE_PIC'] = 'profile_pic.jpg'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Maximum content dimension
