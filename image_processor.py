import cv2

from common.constants import image_size


class ImageProcessor:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ImageProcessor, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def process_screenshot(img):
        height, width = image_size

        # Calculate borders
        if height >= width:
            border_v = int(((height / width) * img.shape[1] - img.shape[0]) / 2)
            border_h = 0
        else:
            border_v = 0
            border_h = int(((width / height) * img.shape[0] - img.shape[1]) / 2)

        # Apply border padding
        img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)

        # Resize the image to the target size
        img = cv2.resize(img, (width, height))

        return img

