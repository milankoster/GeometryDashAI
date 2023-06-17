import time

import cv2
import gd
import mouse
import pandas as pd

from environment import GeometryDashEnvironment
from game_interface import GeometryDashInterface
from common.image_processor import ImageProcessor


class Recorder:
    def __init__(self):
        self.memory = gd.memory.get_memory()
        self.game_interface = GeometryDashInterface()
        self.image_processor = ImageProcessor()

        self.recorded_mouse_states = []
        self.recorded_image_names = []

    def record(self):
        while True:
            self.record_attempt()

    def record_attempt(self):
        env = GeometryDashEnvironment()
        mouse_states = []
        images = []
        image_names = []
        image_counter = len(self.recorded_image_names)

        while True:
            if not self.memory.is_in_level():
                continue
            if not env.has_revived():
                continue

            # Get User Action
            mouse_states.append(mouse.is_pressed())

            # Get Image
            raw_image = self.game_interface.screenshot()
            processed_image = self.image_processor.process_screenshot(raw_image)

            images.append(processed_image)
            image_names.append(f'screenshot-{image_counter}.png')
            image_counter += 1

            if self.memory.is_dead():
                print('Run Failed. Restarting.')
                break

            if self.memory.percent > 99.9:
                self.save_attempt(mouse_states, images, image_names)
                time.sleep(10)
                env.pause()
                break

    def save_attempt(self, mouse_states, images, image_names):
        self.recorded_mouse_states += mouse_states
        self.recorded_image_names += image_names

        df = pd.DataFrame()
        df['mouse_state'] = self.recorded_mouse_states
        df['image_path'] = self.recorded_image_names
        df.to_csv(f'results/imitation-learning.csv')

        for index, image in enumerate(images):
            image_name = image_names[index]
            cv2.imwrite(f'screenshots/{image_name}', image)

        print('Successful run saved.')
