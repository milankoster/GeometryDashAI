import dxcam
import mouse


class GeometryDashInterface:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.camera = dxcam.create(output_idx=0, output_color="GRAY")

    def screenshot(self):
        return self.camera.grab()

    @staticmethod
    def jump():
        mouse.press()

    @staticmethod
    def no_jump():
        mouse.release()
