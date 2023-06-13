import gd

import keyboard as keyboard
import time

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

        reward = self.get_reward(action)

        done = self.memory.is_dead()

        return self.get_state(), reward, done

    def get_reward(self, action):
        reward = 0

        if self.memory.is_dead():
            reward += -100

        if action == Action.JUMP:
            reward += 20

        if self.memory.percent > 2:
            reward += self.memory.percent

        if self.memory.percent > 99:
            reward += 1000

        return reward

    def handle_action(self, action):
        if action == Action.JUMP:
            self.game_interface.jump()
            time.sleep(0.35)
        elif action == Action.NOTHING:
            self.game_interface.no_jump()
        else:
            raise Exception('Invalid Action')

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
