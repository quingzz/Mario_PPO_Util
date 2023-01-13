## Mario PPO Training Util
provide a simplified interface to train Super Mario Bros (NES). Mainly for my own
use later on but you are welcome to use it =))

### Tested packages version
| Packages             | Version |
|----------------------|---------|
| Pytorch              | 1.13.1  |
| Gym                  | 0.21.0  |
| Stable Baseline3     | 1.6.2   |
| gym-super-mario-bros | 7.4.0   | 
Packages tested on python 3.9

### Set up
* Create a conda environment (I use miniforge in this case):  https://github.com/conda-forge/miniforge  
* Install pytorch via conda: https://pytorch.org
* Install required packages via pip by the command:
```commandline
pip install -r requirements.txt
```

### About the project
Two main files in the project are:
* `train_mario.py`: A template for using TrainMarioUtil, run the file if yah wanna go straight to training step 
(it currently using default values so result might not be that good)
* `train_mario_util.py`: Contains all the spaghetti code for training. Take a look at this file to see how the 
environment is processed   

The folder `trained_models` contain several models that trained with some different parameters, all are bad unfortunately 
:// I will eventually get a good model for yah. All models are trained with `1000000` timesteps, non-vectorized environment.

### Important note
If you want to upload your trained model to the repo, remember to `.gitignore` other files/folders except `train_models`

### About TrainMarioUtil 
A documentation (kinna) to use the provided tool
#### Init
```
class TrainMarioUtil(policy="MlpPolicy", model_name="MarioPPO", mario_world="-1-1",log_dir="logs", tensorboard="board", 
learning_rate=0.00003, n_steps=2048, batch_size=512, num_env=1, multiprocess=False, device="auto", n_epochs=8, gamma=0.99,
ent_coef=0.001, clip_range=0.17)
```
* `policy` policy to train the model with 
* `model_name` name of the model, used in saving, logging
* `mario_world` the level in mario to train, default to train at world 1 stage 1. Take a string with format `-<world>-<stage>`
* `log_dir` name of directory to save best model, checkpoints, etc.
* `tensorboard` name of directory to save tensorboard logs
* `learning_rate` model learning rate
* `n_steps` the number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
* `batch_size` minibatch size for training
* `num_env` number of envs to train with. If multiprocessing is applied, make sure it is less than or equal to the number of cores on your computer
* `multiprocess` whether to apply multiprocessing in training (train with multiple envs at once)
* `device` device to train with, use `"cuda"` to train with GPU
* `n_epochs` number of epoch when optimizing the surrogate loss
* `gamma` discount factor
* `ent_coef` entropy coefficient for the loss calculation
* `clip_range`clipping parameter, it can be a function of the current progress remaining (from 1 to 0)

#### Functions
***Train model***
```
train_model(steps=1000000, save_freq=50000, log_freq=1000)
```
Train a model and save it as `model_name` in init
* `steps`: number of steps taken in environment during training
* `save_freq`: frequency to save checkpoint model. Default to save per 50000 steps
* `log_freq`: frequency to log data to tensorboard. Default to save per 1000 steps   

***Continue training model***
```
cont_train(self, steps=1000000, save_freq=50000, log_freq=1000)
```
Continue training the saved model, assuming the model `model_name` in init exist.
The saved model after continue training will be saved with `_CONT` suffix

***Run model***
```
run_model()
```
Run the model `model_name` in init

### Some other important documentations
* Stable baselines 3 documentation for PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
* Documentation for Super Mario Bros environment being used: https://pypi.org/project/gym-super-mario-bros/
