# Include all the imports here
from typing import Dict, Sequence

from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import dmpo
from acme.tf import networks
from acme.tf import utils as tf2_utils
import tensorflow as tf

from datetime import datetime
import imageio
import numpy as np
import sonnet as snt

from examples.offline import bc_robo_utils

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

demo_path = "/home/mohan/research/experiments/bc/panda_lift/expert_demonstrations/1622106811_9832993/demo.hdf5"


def make_environment(env_config, controller_config, keys):
    env_suite = suite.make(**env_config,
                 has_renderer=False,
                 has_offscreen_renderer=False,
                 use_camera_obs=False,
                 reward_shaping=True,
                 controller_configs=controller_config,
                 )
    env = GymWrapper(env_suite, keys=keys)
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    
    spec = specs.make_environment_spec(env)
    
    return env, spec




# Prepare the agent

def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -500.,
    vmax: float = 500.,
    num_atoms: int = 51,
) -> Dict[str, types.TensorTransformation]:
      """Creates networks used by the agent."""

      # Get total number of action dimensions from action spec.
      num_dimensions = np.prod(action_spec.shape, dtype=int)

      # Create the shared observation network; here simply a state-less operation.
      observation_network = tf2_utils.batch_concat

      # Create the policy network.
      policy_network = snt.Sequential([
          networks.LayerNormMLP(policy_layer_sizes),
          networks.MultivariateNormalDiagHead(num_dimensions)
      ])

      # The multiplexer transforms concatenates the observations/actions.
      multiplexer = networks.CriticMultiplexer(
          critic_network=networks.LayerNormMLP(critic_layer_sizes),
          action_network=networks.ClipToSpec(action_spec))

      # Create the critic network.
      critic_network = snt.Sequential([
          multiplexer,
          networks.DiscreteValuedHead(vmin, vmax, num_atoms),
      ])

      return {
          'policy': policy_network,
          'critic': critic_network,
          'observation': observation_network,
      }

def main(_):
    # Prepare the environment
    env_config = {
        "control_freq": 20,
        "env_name": "Lift",
        "hard_reset": False,
        "horizon": 500,
        "ignore_done": False,
        "reward_scale": 1.0,
        "camera_names": "frontview",
        "robots": [
        "Panda"
        ]
    }
    controller_config = load_controller_config(default_controller="OSC_POSE")

    keys = ["object-state"]
    for idx in range(1):
        keys.append(f"robot{idx}_proprio-state")

    env, spec = make_environment(env_config, controller_config, keys)

    agent_networks = make_networks(spec.actions)

    # construct the agent
    agent = dmpo.DistributionalMPO(
        environment_spec=spec,
        checkpoint=True,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        observation_network=agent_networks['observation'],  # pytype: disable=wrong-arg-types
    )

    # agent._learner._checkpointer._time_delta_minutes = 5.

    robot_name = 'Panda'

    eval_env = suite.make(
        **env_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_heights=512,
        camera_widths=512,
        controller_configs=controller_config
    )

    print(f"model loaded successfully")
    eval_steps = 500
    video_path = "/home/mohan/research/experiments/dmpo/panda_lift/eval_rollouts/"

    for run in range(FLAGS.num_episodes):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        cur_path = video_path + f"eval_run_{current_time}_{robot_name}_{run}.mp4"
        # create a video writer with imageio
        writer = imageio.get_writer(cur_path, fps=20)

        full_obs = eval_env.reset()
        flat_obs = np.concatenate([full_obs[key] for key in keys])
        flat_obs = np.float32(flat_obs)
        print(f"obs type is: {flat_obs.dtype}")
        action = agent.select_action(flat_obs)
        print(f"action dtype is: {action.dtype}")
        total_reward = 0
        for i in range(eval_steps):
                # act and observe
                obs, reward, done, _ = eval_env.step(action)
                # eval_env.render()
                total_reward += reward
                # compute next action
                flat_obs = np.concatenate([obs[key] for key in keys])
                action = agent.select_action(np.float32(flat_obs))

                # dump a frame from every K frames
                if i % 1 == 0:
                    frame = obs["frontview_image"]
                    frame = np.flip(frame, 0)
                    writer.append_data(frame)
                if done:
                    break
        print(f"total eval reward: {total_reward}")

    # Start the training process
    
    # loop = acme.EnvironmentLoop(env, agent)

    # num_episodes = FLAGS.num_episodes
    # loop.run(num_episodes=num_episodes)

if __name__ == '__main__':
  app.run(main)



