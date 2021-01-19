import datetime as dt
import gym
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

env = gym.make("BreakoutNoFrameskip-v4")


def to_grayscale(state, width, height):
    state = state[20:]  # Cut off score
    state = np.sum(state, axis=2) / 255  # Combine color channels
    state = cv2.resize(state, (width, height))
    return state


def create_model():
    model = models.Sequential()
    model.add(layers.Input((84, 84, 4)))
    model.add(layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(4, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss=keras.losses.Huber()
    )
    return model

model = create_model()
target_model = create_model()

# Uncomment and complete filename to continue training from file
# model = models.load_model('models/xxxx-xx-xx-xx-xx-xx')
# target_model.set_weights(model.get_weights())

discount = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_steps = 500000
epsilon_decay = (epsilon - epsilon_min) / epsilon_steps
batch_size = 32

state_history = []
episode_rewards = []
episode_count = 0
frame_count = 0

max_memory_size = 100000
update_target_network_frames = 10000

while True:  # Run until solved
    state = np.array(env.reset(), dtype="float")
    episode_reward = 0
    state = to_grayscale(state, 84, 84)

    # Stacking 4 frames
    state = np.stack((state, state, state, state), axis=2)

    done = False
    while not done:
        frame_count += 1
        if frame_count < 50000 or epsilon > np.random.rand(1)[0]:
            # Exploration
            action = np.random.choice(4)
            if epsilon > epsilon_min:
                epsilon -= epsilon_decay
        else:
            # Exploitation
            action = np.argmax(model.predict(np.array([state])))

        # Apply step
        next_state, reward, done, info = env.step(action)

        next_state = to_grayscale(next_state, 84, 84)
        next_state = np.stack((state[:, :, 1], state[:, :, 2], state[:, :, 3], next_state), axis=2)

        episode_reward += reward

        state_history.append((state, action, reward, done))

        state = next_state

        # Fit model every 4 frames
        if frame_count % 4 == 0 and len(state_history) - 1 > batch_size:

            # Select minibatch frames
            i_list = np.random.choice(range(len(state_history) - 1), size=batch_size)

            state_sample = np.array([state_history[i][0] for i in i_list])
            next_state_sample = np.array([state_history[i+1][0] for i in i_list])
            reward_sample = np.array([state_history[i][2] for i in i_list])
            done_sample = np.array([float(state_history[i][3]) for i in i_list])

            # Update q_values
            future_rewards = target_model.predict(next_state_sample)

            q_values = reward_sample + discount * np.amax(future_rewards, axis=1)

            q_values = q_values * (1 - done_sample) - done_sample

            # Fit the model on the selected frames
            model.fit(state_sample, q_values, verbose=0)

        if frame_count % update_target_network_frames == 0:
            target_model.set_weights(model.get_weights())
            date = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            model.save('./models/model-{}'.format(date), save_format="tf")

        if len(state_history) > max_memory_size:
            del state_history[0]

        if info["ale.lives"] == 0:
            done = True
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > 100:
                del episode_rewards[0]
            episode_count += 1

        if frame_count % 1000 == 0:
            print("frame {}, episode {}, reward: {}".format(
                frame_count, episode_count, np.mean(episode_rewards)))
