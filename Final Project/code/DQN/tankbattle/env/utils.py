# from scipy.misc import imresize
import PIL
from PIL import Image
import time
import numpy as np
import torch

# Utilities

class Utils():
    WHITE = 0
    BLACK = 1
    GRAY = 2

    @staticmethod
    def get_current_time():#Get current time
        return int(round(time.time()))

    @staticmethod
    def get_color(color):#Colorize
        if color == Utils.WHITE:
            return (255, 255, 255)
        elif color == Utils.BLACK:
            return (0, 0, 0)
        elif color == Utils.GRAY:
            return (80, 80, 80)

    @staticmethod
    def process_state(state):
        """
        Define the function process_state to preprocess the input raw image.
        It takes a raw image as input, an RGB image of size (height, width, 3).
        It outputs a grayscale and resized image of size (84, 84).
        :param state:
        :return:
        """
        grayscale = np.dot(state[:, :, :3], [0.299, 0.587, 0.114])
        resize = np.array(Image.fromarray(grayscale).resize((84, 84),resample=PIL.Image.BILINEAR))
        return resize

    @staticmethod
    def resize_image(image):
        # Convert RGB image to PIL image object
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        # Resize to 84x84 pixels
        resized_image = pil_image.resize((84, 84), resample=Image.BILINEAR)
        # Convert image to NumPy array
        resized_image = np.array(resized_image)
        # Transpose the shape of the array from (height, width, 3) to (3, height, width)
        resized_image = np.transpose(resized_image, (2, 0, 1))
        return resized_image

    @staticmethod
    def transpose_image(image):
        """
        Convert image format from (height, width, in_channels) to (in_channels, height, width)
        :param image: Input image, ndarray format, shape (height, width, in_channels)
        :return: Transformed image, ndarray format, shape (in_channels, height, width)
        """
        return np.transpose(image, (2, 0, 1))

    @staticmethod
    def save_model(model, param_save_path, episode, epsilon, reward):
        # Save our model
        torch.save(model.state_dict(), f"{param_save_path}predicted_{episode}_{epsilon}_{reward}.pth")

    @staticmethod
    def load_model(model, path):
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def data_saving_format(data):
        str = ""
        for item in data:
            str += f"{item}\n"
        return str

    @staticmethod
    def save_data(data, path):
        # Save data to a txt file under the path
        with open(path, 'w') as f:
            f.write(str(data))
