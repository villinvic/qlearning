import fire
import env


def RUN(env_id='LunarLander-v2', lr=5e-4, gamma=0.99, epsilon_decay=0.996, epsilon_end=0.01):
    player = env.Player(env_id, lr, gamma, epsilon_decay, epsilon_end)
    player.main_loop()


if __name__ == '__main__':
    fire.Fire(RUN)