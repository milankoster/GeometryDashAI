import dxcam
import mouse


class GeometryDashInterface:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.camera = dxcam.create(output_idx=0)
        return cls._instance

    def screenshot(self):
        image = None

        # camera.grab can very occasionally return None
        while image is None:
            image = self.camera.grab()

        return image

    @staticmethod
    def jump():
        mouse.click()

    @staticmethod
    def no_jump():
        mouse.release()
