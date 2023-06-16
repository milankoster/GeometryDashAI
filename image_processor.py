import cv2

from common.constants import *


class ImageProcessor:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def process_screenshot(img):
        # Calculate borders
        if image_height >= image_width:
            border_v = int(((image_height / image_width) * img.shape[1] - img.shape[0]) / 2)
            border_h = 0
        else:
            border_v = 0
            border_h = int(((image_width / image_height) * img.shape[0] - img.shape[1]) / 2)

        # Apply border padding
        img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Resize the image to the target size
        img = cv2.resize(img, (image_width, image_height))

        return img

