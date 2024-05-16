from typing import Any, Callable, Tuple

import jax.numpy as jnp
import qdax
from qdax import environments
from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdax.types import Genotype, RNGKey


def get_environment_brax(
    env_name: str,
    episode_length: int,
    fixed_init_state: bool,
) -> Any:

    # Initialising environment
    if env_name == "anttrap":
        env = environments.create(
            env_name,
            episode_length=episode_length,
            fixed_init_state=fixed_init_state,
            use_contact_forces=False,
            exclude_current_positions_from_observation=False,
        )
    if env_name == "ant_uni":
        env = environments.create(
            env_name,
            episode_length=episode_length,
            fixed_init_state=fixed_init_state,
            use_contact_forces=False,
        )
    else:
        env = environments.create(
            env_name,
            episode_length=episode_length,
            fixed_init_state=fixed_init_state,
        )

    return env


def get_policy_struc_brax(
    env: Any,
    policy_hidden_layer_sizes: Tuple,
) -> Tuple[int, int, Tuple, Callable]:
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    return env.action_size, env.observation_size, policy_layer_sizes, jnp.tanh


def get_reward_offset_brax(env: Any, env_name: str) -> jnp.ndarray:
    return environments.reward_offset[env_name]


def get_behavior_descriptor_length_brax(env: Any, env_name: str) -> jnp.ndarray:
    return env.behavior_descriptor_length


def get_scoring_function_brax(
    env: Any,
    env_name: str,
    episode_length: int,
    policy_network: Genotype,
    random_key: RNGKey,
) -> Callable:

    bd_extraction_fn = qdax.environments.behavior_descriptor_extractor[env_name]
    scoring_fn, random_key = create_brax_scoring_fn(
        env,
        policy_network,
        bd_extraction_fn,
        random_key,
        episode_length=episode_length,
    )
    return scoring_fn, random_key
