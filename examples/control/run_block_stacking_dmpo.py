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

import numpy as np
import sonnet as snt

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
        "env_name": "Stack",
        "hard_reset": False,
        "horizon": 500,
        "ignore_done": False,
        "reward_scale": 1.0,
        "camera_names": "frontview",
        "robots": [
        "Panda"
        ]
    }
    controller_config = load_controller_config(default_controller="OSC_POSITION")

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
    agent._learner._checkpointer._time_delta_minutes = 10.
    # Start the training process
    loop = acme.EnvironmentLoop(env, agent)

    num_episodes = FLAGS.num_episodes
    loop.run(num_episodes=num_episodes)

if __name__ == '__main__':
  app.run(main)



