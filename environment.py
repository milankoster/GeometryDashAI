import gd
import math

import keyboard as keyboard

from common.action import Action
from common.constants import image_size
from game_interface import GeometryDashInterface
from image_processor import ImageProcessor


class GeometryDashEnvironment:
    def __init__(self):
        self.memory = gd.memory.get_memory()
        self.game_interface = GeometryDashInterface()
        self.image_processor = ImageProcessor()

        self.action_space = 2
        self.state_space = image_size

        self.highest_rounded_percent = 0
        self.revived = False

    def step(self, action):
        self.handle_action(action)

        if self.memory.is_dead():
            reward = -1000
        elif action == Action.JUMP:
            reward = -1
        elif self.percentage_improved():
            reward = 10
        else:
            reward = 0

        done = self.memory.is_dead()

        return self.get_state(), reward, done

    def handle_action(self, action):
        if action == Action.JUMP:
            self.game_interface.jump()
        elif action == Action.NOTHING:
            self.game_interface.no_jump()
        else:
            raise Exception('Invalid Action')

    def percentage_improved(self):
        current_percent = self.memory.percent
        rounded_percent = math.floor(current_percent)

        if rounded_percent > self.highest_rounded_percent:
            self.highest_rounded_percent = rounded_percent
            return True
        return False

    def get_state(self):
        raw_image = self.game_interface.screenshot()
        if raw_image is None:
            return None

        processed_image = self.image_processor.process_screenshot(raw_image)
        return processed_image

    def has_revived(self):
        if not self.memory.is_dead():
            self.revived = True
        return self.revived

    @staticmethod
    def pause():
        keyboard.press_and_release('escape')

    @staticmethod
    def unpause():
        keyboard.press_and_release('space')
