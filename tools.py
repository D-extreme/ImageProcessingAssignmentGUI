# Author: Shashwat Pathak

# This file contains the tools class which contains all the functions to various processes
# that will be used in the image editor

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

class ToolBox(object):
    def __init__(self, input_image):
        self.tools = ['Gaussian Blur', 'Histogram Equalization', 'Log Transform', 'Gamma Transform', 'Median Filter']
        self.image = input_image

    def display_image(self):
        plt.imshow(self.image)

    def rgb2gray(self):
        return np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140]) # Code adapted from @stackoverflow https://stackoverflow.com/a/12201744

    def gaussian_blur(self,size,sigma):
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1] # Code adapter from @stackoverflow https://stackoverflow.com/a/27928469
        filt = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        filt  = filt / filt.sum()
        blurred_image = sig.convolve(self.image, filt, mode='same')
        return blurred_image

    def histogram_equalization(self, implementation_type, window_size):
        arr = self.image.flatten()
        bins = 256

        # Getting the cumulative histogram
        n, bins, patches = plt.hist(arr, bins, cumulative=True)
        plt.close()

        # Normalizing the histogram
        n_normalized = n * (1 / np.size(self.image))

        # Getting the transformed histogram after equalization
        n_final = n_normalized * (len(bins) - 1)
        transformation = np.floor(n_final)
        transformation.astype(np.uint8)

        temp = np.copy(self.image)
        trans_image = np.zeros(np.shape(self.image))
        for i in range(len(temp)):
            for j in range(len(temp[i])):
                trans_image[i][j] = transformation[self.image[i][j]]

        return trans_image

    def log_transform(self):
        return np.log(1 + self.image)

    def gamma_transformation(self, gamma):
        return self.image ** gamma

    def median_filter(self, window_size):









