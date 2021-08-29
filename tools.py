# Author: Shashwat Pathak

# This file contains the tools class which contains all the functions to various processes
# that will be used in the image editor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv as rgb2hsv
from matplotlib.colors import hsv_to_rgb as hsv2rgb
import sklearn
from sklearn.feature_extraction.image import extract_patches_2d


# The toolbox contains all the tools we will be requiring for image enhancements
# The toolbox class creates toolbox objects which can be used for display, image enhancement, conversion, etc

class ToolBox(object):
    def __init__(self, input_image):
        self.tools = ['Gaussian Blur', 'Histogram Equalization', 'Log Transform', 'Gamma Transform', 'Median Filter']
        self.image = input_image  # The input image

    def display_image(self):  # Function for displaying the image
        if (len(self.image.shape) == 3):
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap='gray')

    def rgb2hsv(self):  # function for converting from rgb to hsv
        return rgb2hsv(self.image)

    def hsv2rgb(self):  # function for converting from hsv to rgb
        return hsv2rgb(self.image)

    def rgb2gray(self): # function for converting from rgb to gray
        return np.dot(self.image[..., :3],[0.2989, 0.5870, 0.1140])  # Code adapted from @stackoverflow https://stackoverflow.com/a/12201744

    # function for performing gaussian blur
    def gaussian_blur(self, sigma, kernel_size=3):

        def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
            # Code Adapted from: https://gist.github.com/AxelThevenot/43bb743eeb2836444ce36805c3316094

            # compute 1 dimension gaussian
            gaussian_1d = np.linspace(-1, 1, k)
            # compute a grid distance from center
            x, y = np.meshgrid(gaussian_1d, gaussian_1d)
            distance = (x ** 2 + y ** 2) ** 0.5

            # compute the 2 dimension gaussian
            gaussian_2d= np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
            gaussian_2d = gaussian_2d / (2 * np.pi * sigma ** 2)

            # normalize part (mathematically)
            if normalize:
                gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
            return gaussian_2d

        def convolve2d(image, filter, mode='same'):
            padded_image = []
            sp_x, sp_y = np.shape(filter) # Shape of the filter

            if (mode == 'same'):
                padded_image = np.pad(image, pad_width=np.shape(filter)[0] // 2) # padded image such that the
                # convolution output has the same dimensions as the input

            # Extracting patches from the padded image to vectorize the processing
            patches = extract_patches_2d(padded_image, (sp_x, sp_y), max_patches=None, random_state=None)
            print(len(patches))

            # Function for dot multiplication of matrices and their sum
            def multiply(patch, filter):
                output = np.sum(np.multiply(patch, filter))
                return output

            multiplied_list = list(map(lambda x: multiply(x, filter), patches))

            # Reshaping the converted array after output from the multiplied list
            converted_array = np.expand_dims(multiplied_list, axis=0)
            output = np.reshape(converted_array, np.shape(image))
            return output

        # Converting rgb image to hsv for extracting and processing the v channel
        if (len(self.image.shape) == 3):
            image1 = rgb2hsv(self.image)
            image = image1[:, :, 2].astype('uint8')
        else:
            image = self.image.astype('int32')

        if kernel_size % 2 == 0 and kernel_size != 1:
            kernel_size = kernel_size-1
        gauss = get_gaussian_kernel(kernel_size, sigma)
        trans_image = convolve2d(image, gauss, mode='same')

        # function for converting the image back to rgb after processing
        if len(self.image.shape) == 3:
            image1[:, :, 2] = trans_image
            final = hsv2rgb(image1)
        else:
            final = trans_image

        return final.astype('uint8')

    def histogram_equalization(self):
        # Converting rgb image to hsv for extracting and processing the v channel
        if (len(self.image.shape) == 3):
            image1 = rgb2hsv(self.image)
            image = image1[:, :, 2].astype('uint8')
        else:
            image = self.image.astype('uint8')  # here the input image has intensities in the range 0 to 255

        arr = image.flatten()
        bins = 256

        # Getting the cumulative histogram
        n, bins, patches = plt.hist(arr, bins, cumulative=True)
        plt.close()

        # Normalizing the historgram
        n_normalized = n * (1 / np.size(image))

        # Getting the transformed histogram after equalization
        n_final = n_normalized * (len(bins) - 1)
        transformation = np.floor(n_final)
        transformation.astype(np.uint8)

        def transform(transformation_matrix, x):
            return transformation_matrix[x]

        # Mapping the equalized histogram values back in the array
        output = list(map(lambda x: transform(transformation, x), image.flatten()))
        trans_image = np.reshape(output, (image.shape[0], image.shape[1]))

        # function for converting the image back to rgb after processing
        if (len(self.image.shape) == 3):
            image1[:, :, 2] = trans_image
            final = hsv2rgb(image1)
        else:
            final = trans_image

        return final.astype('uint8')

    def log_transform(self):
        # Converting rgb image to hsv for extracting and processing the v channel
        if (len(self.image.shape) == 3):
            image1 = rgb2hsv(self.image)
            image = image1[:, :, 2]
        else:
            image = self.image.astype('uint8')

        # Getting the scale factor to bring the transformed image in [0,255] range
        c = 255 / np.log(1 + np.max(image))
        # Getting the final transformed image
        transformed_image = c * np.log(1 + image)

        # function for converting the image back to rgb after processing
        if (len(self.image.shape) == 3):
            image1[:, :, 2] = transformed_image.astype('uint8')
            final = hsv2rgb(image1).astype('uint8')
        else:
            final = transformed_image
        return final.astype('uint8')

    def gamma_transformation(self, gamma):
        # Converting rgb image to hsv for extracting and processing the v channel
        if (len(self.image.shape) == 3):
            image1 = rgb2hsv(self.image)
            image = image1[:, :, 2]
        else:
            image = self.image.astype('uint8')

        # Getting the scale factor to bring the transformed image in [0,255] range
        c = 255 / (np.max(image) ** gamma)
        # Getting the final transformed image
        transformed_image = (image ** gamma)
        transformed_image = c * (transformed_image)

        # function for converting the image back to rgb after processing
        if (len(self.image.shape) == 3):
            image1[:, :, 2] = transformed_image.astype('uint8')
            final = hsv2rgb(image1)
        else:
            final = transformed_image

        return final.astype('uint8')

    def sharpening(self, alpha, beta):

        def convolve2d(image, filter, mode='same'):
            padded_image = []
            sp_x, sp_y = np.shape(filter)
            if (mode == 'same'):
                padded_image = np.pad(image, pad_width=np.shape(filter)[0] // 2)

            patches = extract_patches_2d(padded_image, (sp_x, sp_y), max_patches=None, random_state=None)

            def multiply(patch, filter):
                output = np.sum(np.multiply(patch, filter))
                return output

            multiplied_list = list(map(lambda x: multiply(x, filter), patches))
            converted_array = np.expand_dims(multiplied_list, axis=0)
            output = np.reshape(converted_array, np.shape(image))
            return output

        def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
            # Code Adapted from: https://gist.github.com/AxelThevenot/43bb743eeb2836444ce36805c3316094

            # compute 1 dimension gaussian
            gaussian_1d = np.linspace(-1, 1, k)
            # compute a grid distance from center
            x, y = np.meshgrid(gaussian_1d, gaussian_1d)
            distance = (x ** 2 + y ** 2) ** 0.5

            # compute the 2 dimension gaussian
            gaussian_2d= np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
            gaussian_2d = gaussian_2d / (2 * np.pi * sigma ** 2)

            # normalize part (mathematically)
            if normalize:
                gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
            return gaussian_2d

        # Converting rgb image to hsv for extracting and processing the v channel
        if (len(self.image.shape) == 3):
            image1 = rgb2hsv(self.image)
            image_proc = image1[:, :, 2]
        else:
            image_proc = self.image.astype('int32')

        laplacian = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
        gaussian_kernel = get_gaussian_kernel()
        blurred_image = convolve2d(image_proc, gaussian_kernel, mode='same')
        sharpened_output = convolve2d(blurred_image, laplacian, mode='same')
        sharpened_output = (255/np.max(abs(sharpened_output)))*sharpened_output
        sharpened_output = sharpened_output*(sharpened_output > 0)
        output = alpha * (image_proc) + beta * (sharpened_output)

        # function for converting the image back to rgb after processing
        if (len(self.image.shape) == 3):
            image1[:, :, 2] = output
            final = hsv2rgb(image1).astype('uint8')
        else:
            final = output

        return final

    def median_filtering(self, window_size):
        if len(self.image.shape) == 3:
            image1 = rgb2hsv(self.image)
            image_proc = image1[:, :, 2].astype('uint8')
        else:
            image_proc = self.image.astype('int32')

        if window_size % 2 == 0 and window_size != 1:
            window_size = window_size-1

        padded_image = np.pad(image_proc, pad_width=window_size // 2)

        patches = extract_patches_2d(padded_image, (window_size, window_size), max_patches=None, random_state=None)
        replaced_patches = list(map(np.median, patches))
        converted_array = np.expand_dims(replaced_patches, axis=0)
        filtered_image = np.reshape(converted_array, np.shape(image_proc))

        if len(self.image.shape) == 3:
            image1[:, :, 2] = filtered_image.astype('uint8')
            final = hsv2rgb(image1).astype('uint8')
        else:
            final = filtered_image
        # output = converted_array
        return final



