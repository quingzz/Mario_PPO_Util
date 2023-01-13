import train_mario_util
from train_mario_util import TrainMarioUtil


def main():
    mario_train = TrainMarioUtil(model_name="Mario_PPO_ENT_001")
    mario_train.train_model()
    # mario_train.cont_train(save_freq=100000)
    # mario_train.run_model()


if __name__ == "__main__":
    main()
