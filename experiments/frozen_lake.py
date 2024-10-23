import gymnasium as gym

def init_env(render_mode: str = "") -> gym.Env:
    gym.envs.register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )
    
    return gym.make('FrozenLakeNotSlippery-v0', render_mode=render_mode)
