import gc
import os
import random
from collections import deque

import absl.logging
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from common.action import Action
from common.constants import *
from environment import GeometryDashEnvironment


class Trainer:
    def __init__(self):
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00025
        self.batch_size = 128

        self.episodes = 2000

        self.env = GeometryDashEnvironment()

        self.memory = deque(maxlen=2500)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_counter = 0
        self.target_update_freq = 3

        # Logging values per run
        self.percentages = []
        self.rewards = []

        self.attempts = []
        self.jumps = []
        self.total_jumps = self.env.memory.jumps

        # Remove unintended warnings during training
        absl.logging.set_verbosity(absl.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, image_dimensions)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def act(self, state, evaluate=False):
        if np.random.rand() <= self.epsilon and not evaluate:
            return random.randrange(self.env.action_space)
        act_values = self.target_model.predict(state, verbose=0)
        act_values = act_values[0]

        return np.argmax(act_values)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Generate the target values using the target model
        target_q_values = self.target_model.predict_on_batch(next_states)
        targets = rewards + self.gamma * np.amax(target_q_values, axis=1) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.update_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.update_counter += 1

    def train(self, model_name):
        for episode in range(1, self.episodes + 1):
            gc.collect()
            self.env = GeometryDashEnvironment()

            current_state = self.env.get_state()
            current_state = np.reshape(current_state, (1, image_width, image_height, image_dimensions))

            epi_reward = 0

            while True:
                if not self.env.memory.is_in_level():
                    continue
                if not self.env.has_revived():
                    continue

                action = self.act(current_state)
                new_state, reward, done = self.env.step(Action(action))

                new_state = np.reshape(new_state, (1, image_width, image_height, image_dimensions))

                epi_reward += reward

                self.remember(current_state, action, reward, new_state, done)

                current_state = new_state

                if done:
                    self.env.pause()
                    self.replay()

                    self.log_episode(episode, epi_reward)
                    self.save_model(model_name, episode)
                    self.save_logs(model_name)
                    self.env.unpause()

                    if episode % 10 == 0:
                        self.evaluate()
                    break

    def log_episode(self, episode, epi_reward):
        self.attempts.append(self.env.memory.attempts)
        self.percentages.append(self.env.memory.percent)
        self.rewards.append(epi_reward)

        episode_jumps = self.env.memory.jumps - self.total_jumps
        self.total_jumps = self.env.memory.jumps
        self.jumps.append(episode_jumps)

        print(f'episode: {episode}/{self.episodes}, level progress: {self.env.memory.percent}, '
              f'reward: {epi_reward}, jumps: {episode_jumps}, epsilon: {self.epsilon}')

    def save_logs(self, model_name):
        df = pd.DataFrame()
        df['Attempt'] = self.attempts

        df['Percentages'] = self.percentages
        df['Rewards'] = self.rewards

        df['Jumps'] = self.jumps

        if not os.path.exists('results'):
            os.mkdir('results')
        df.to_csv(f'results/{model_name}.csv')

    def save_model(self, model_name, episode):
        should_save = False

        if episode < 10:
            should_save = True
        elif 10 <= episode < 200 and episode % 10 == 0:
            should_save = True
        elif episode % 50 == 0:
            should_save = True

        if should_save:
            file_name = f'models/{model_name}/episode-{episode}.model'
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            self.model.save(file_name)

    def evaluate(self):
        gc.collect()
        self.env = GeometryDashEnvironment()

        current_state = self.env.get_state()
        current_state = np.reshape(current_state, (1, image_width, image_height, image_dimensions))

        epi_reward = 0

        while True:
            if not self.env.memory.is_in_level():
                continue
            if not self.env.has_revived():
                continue

            action = self.act(current_state, evaluate=True)

            new_state, reward, done = self.env.step(Action(action))

            current_state = np.reshape(new_state, (1, image_width, image_height, image_dimensions))

            epi_reward += reward

            if done:
                episode_jumps = self.env.memory.jumps - self.total_jumps

                print(f'EVALUATION: level progress: {self.env.memory.percent}, '
                      f'reward: {epi_reward}, jumps: {episode_jumps}')
                break


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train('base')
