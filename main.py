import keras.models

from trainer import Trainer

if __name__ == '__main__':
    game_controller = Trainer()

    model = keras.models.load_model('models/imitation-learning-v1.h5')
    game_controller.model = model

    game_controller.evaluate()
