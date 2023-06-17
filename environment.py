import gd
import time
import keyboard as keyboard

from common.action import Action
from common.constants import *
from game_interface import GeometryDashInterface
from common.image_processor import ImageProcessor


class GeometryDashEnvironment:
    def __init__(self):
        self.memory = gd.memory.get_memory()
        self.game_interface = GeometryDashInterface()
        self.image_processor = ImageProcessor()

        self.action_space = 2
        self.state_space = (image_width, image_height)

        self.revived = False

    def step(self, action):
        self.handle_action(action)

        reward = 0

        reward += self.memory.percent * 10

        done = self.memory.is_dead()

        return self.get_state(), reward, done

    def handle_action(self, action):
        if action == Action.JUMP:
            self.game_interface.jump()
            time.sleep(sleep_duration)
        elif action == Action.NOTHING:
            self.game_interface.no_jump()
        else:
            raise Exception('Invalid Action')

    def get_state(self):
        raw_image = self.game_interface.screenshot()

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
