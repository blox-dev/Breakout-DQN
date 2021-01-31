import cv2
import gym
import time
import numpy as np
from tensorflow.keras import models

env = gym.make("BreakoutNoFrameskip-v4")

model = models.load_model('models/model-2021-01-31-21-32-50')

episode_rewards = []
episode_count = frame_count = 0


def to_grayscale(state, width, height):
    state = state[20:]  # Cut off score
    state = np.sum(state, axis=2) / 255  # Combine color channels
    state = cv2.resize(state, (width, height))
    return state


while True:  # Run until solved
    state = np.array(env.reset(), dtype="float")
    state = to_grayscale(state, 84, 84)

    # Stacking 4 frames
    state = np.stack((state, state, state, state), axis=2)

    done = False
    episode_reward = 0
    env.step(1)
    while not done:
        # Render the environment
        env.render()
        time.sleep(1/30)

        frame_count += 1
        action = np.argmax(model.predict(np.array([state])))

        # Apply step
        next_state, reward, done, info = env.step(action)
        next_state = to_grayscale(next_state, 84, 84)
        next_state = np.stack((state[:, :, 1], state[:, :, 2], state[:, :, 3], next_state), axis=2)

        state = next_state

        if info["ale.lives"] == 0:
            done = True
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > 100:
                del episode_rewards[0]
            episode_count += 1

        if frame_count % 1000 == 0:
            print("frame {}, episode {}, reward: {}".format(
                frame_count, episode_count, np.mean(episode_rewards)))
