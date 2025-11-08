import gymnasium


def make_environment(name: str):
    env = gymnasium.make(name, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env, "./videos", lambda episode: episode % 50 == 0
    )
    return env
