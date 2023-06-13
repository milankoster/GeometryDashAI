import dxcam
import mouse


class GeometryDashInterface:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.camera = dxcam.create(output_idx=0, output_color="GRAY")
        return cls._instance

    def screenshot(self):
        return self.camera.grab()

    @staticmethod
    def jump():
        mouse.click()

    @staticmethod
    def no_jump():
        mouse.release()
