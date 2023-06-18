import keras.models

from trainer import Trainer

if __name__ == '__main__':
    imitation_model = keras.models.load_model('models/imitation-learning-v1.h5')

    evaluator = Trainer()
    evaluator.model = imitation_model

    while True:
        evaluator.evaluate()
