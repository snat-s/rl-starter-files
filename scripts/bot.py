"""
This was originaly a test but it can be used to generate a dataset.
The original is from here: https://github.com/Farama-Foundation/Minigrid/blob/master/tests/test_baby_ai_bot.py

Then I used this as inspiration to generate datasets for each level: https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation/blob/main/scripts/minigrid/generate_minigrid.py

To run this you are going to need both, minari and minigrid cloned because I
am using stuff that has not shipped to the packages.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces.text import alphanumeric
import minari
import minigrid
from minari import DataCollector
from minigrid.utils.baby_ai_bot import BabyAIBot

# see discussion starting here: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
        "BabyAI-PutNextS5N2Carrying-v0",
        "BabyAI-PutNextS6N3Carrying-v0",
        "BabyAI-PutNextS7N4Carrying-v0",
        "BabyAI-KeyInBox-v0",
        }

# on a final run this should be 100_000 episodes
TOTAL_EPISODES = 1_000 #100_000 

# get all babyai envs (except the broken ones)
babyai_envs = []
for k_i in gym.envs.registry.keys():
    if k_i.split("-")[0] == "BabyAI":
        if k_i not in broken_bonus_envs:
            babyai_envs.append(k_i)

def generate_dataset(env_id):
    # Use the parameter env_id to make the environment
    #env = gym.make(env_id, render_mode="human") # for visual debugging
    env = gym.make(env_id)
    max_len = 999 
    #print(env.observation_space)
    obs_space = gym.spaces.Dict({
        "direction": env.observation_space["direction"],
        "image": env.observation_space["image"],
        "mission": gym.spaces.Text(
            max_length=max_len,
            charset=str(alphanumeric) + ' ',
            )
        })

    env = DataCollector(env, record_infos=True, max_buffer_steps=1_000_000, observation_space=obs_space)

    # reset env
    curr_seed = 0

    for _ in range(TOTAL_EPISODES):
        env.reset(seed=curr_seed)

        expert = BabyAIBot(env)

        last_action = None
        env.render()

        while True:
            #action = expert.replan(last_action)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            last_action = action
            env.render()

            if terminated or truncated:
                break

        # try again with a different seed
        curr_seed += 1

    dataset = env.create_dataset(dataset_id=f"{env_name}_example",
                                 algorithm_name="Bot",
                                 code_permalink="nonrightnow",
                                 author="ManifoldRG",
                                 author_email="blabla@gmail.com")
    env.close()

for env_name in babyai_envs:
    generate_dataset(env_name)
