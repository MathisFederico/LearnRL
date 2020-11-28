# Cartpole example

## Installation

To get started, you will need Python 3.5+, LearnRL and TensorFlow (v2.2 for Python 3.8) installed. Simply install them using `pip` :

```python
pip install learnrl tensorflow
```

More information for installing TensorFlow, [here](https://www.tensorflow.org/install/pip?hl=fr).

## Gym environments

In Reinforcement Learning, Gym environments are often used since they offer a simple and common framework to work with and introduce reproducibility.

Don't hesitate to check :
-   [Available Gym environments](https://gym.openai.com/envs/#classic_control)
-   [Gym Documentation](https://gym.openai.com/docs/)

In this example, we will be workig with the `CartPole-v1` environment. Let's create the environment :

```python
import gym

env = gym.make('CartPole-v1')
env.reset()

for _ in range(1000):
  env.render()
  env.step(env.action_space.sample()) # Take a random action

env.close()
```

`env = gym.make('CartPole-v1')` creates the environment. `env` have the following methods, which are pretty straightforward :
-   `env.render()` : Render the environment.
-   `env.reset()` : Reset it.
-   `env.step(<action>)` : Update the environment when the agent does `<action>`. Here we use `env.action_space.sample()` to do a random action each step.

`env` also offers the following attributes : `env.observation_space` and `env.action_space`, which describe the observation and action space (`Discrete` and `Box` are the main types of space).

For now, we are lacking some information to write our agent, such as the observation and reward at each step ! In reality, these information are returned by `env.step(<action>)`! We have access to them thanks to :

```python
observation, reward, done, info = env.step(action)
```

-   `observation` : data representing our environment, whose shape is given by `env.observation_space`.
-   `reward` (*float*) : the reward achieved by the last action.
-   `done` (*boolean*) : `True` if the episode has terminated and if it is time to reset.
-   `info` (*dist*) : More information, mainly for debugging or logging.

## Our first agent

Let's create our first agent using learnrl.Agent, which provides us a simple API to build agent ([Documentation](https://learnrl.readthedocs.io/en/latest/core.html#agent)). To get started, we just need to define the `act` method.

```python
import learnrl as rl
import numpy as np

class RandomAgent(rl.Agent):
  def act(self, observation, greedy=False):
    return np.random.choice([0, 1])
```

We can now use it with :
 
```python
n_episodes = 5 # Run for 5 episodes
rd_agent = RandomAgent()

for i in range(n_episodes):
  done = False
  observation = env.reset()
  while not done:
    env.render()
    action = rd_agent.act(observation)
    observation, reward, done, info = env.step(action)

  print(f"Episode {i} / {n_episodes}")
```

This piece of code is quite common in reinforcement learning and doesn't depend of the environment or the agent. Hence, LearnRL provides the `Playground` class.

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-random_agent-py)

The previous code can be replaced by :

```python
rd_agent = RandomAgent()
playground = rl.Playground(env, rd_agent)

playground.test(5, verbose=1)
```

`Playground` provides the following methods :
-   `test(episodes, ...)` : Test the agent on the environment for a number of episodes.
-   `fit(episodes, ...)` : Train the agent (calling its learn method) for a number of episodes.

The `verbose` argument allows to set logging level : 0 (Silent, no logging), 1 (Episodes cycles), 2 (Episodes), 3 (Steps cycles), 4 (Steps), 5 (Detailed steps).

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-random_agent_playground-py)

## Deep Q Learning

It is time to get serious and create our first agent, who can learn from its mistake and success.

We are going to use the `StandardAgent` API from LearnRL ([Documentation](https://learnrl.readthedocs.io/en/latest/agents/standard.html)). It provides the default Q-Learning Evaluation and epsilon-Greedy control, we just need to provide an Estimator to the [DeepRLAgent](https://learnrl.readthedocs.io/en/latest/agents/standard.html#deeprlagent).

```python
import tensorflow as tf
from tensorflow.keras import models, optimizers, layers

class MyEstimator(rl.estimators.KerasEstimator):
  def build(self):
    self.model = models.Sequential()
    self.model.add(layers.Flatten(input_shape=self.observation_shape))
    self.model.add(layers.Dense(self.observation_size, activation='tanh'))
    self.model.add(layers.Dense(self.action_size, activation='linear'))

    self.model.compile(optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

  def preprocess(self, observations:tf.Tensor, actions:tf.Tensor=None):
    return observations

custom_action_value = MyEstimator(observation_space=env.observation_space,
                                  action_space=env.action_space,
                                  learning_rate=1e-2)
```

Then, we just have to create our agent and use this estimator :

```python
from learnrl.agents import StandardAgent

agent = StandardAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      action_values=custom_action_value)

playground = rl.Playground(env, agent)
playground.fit(500, verbose=1)
playground.test(5, verbose=1)
```

There should be no render during training, but we will see how well our agent performs using `test`.

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-standard_agent-py)

## Normalization

Our agent can access very high rewards, the maximum being 500 in the cartpole environment. But high values are harder to predict for a neural network, hence we are going to normalize the reward with `learnrl.RewardHandler`.

Let's just divide the reward by 500.0.

```python
class RewardScaler(rl.RewardHandler):
  def reward(self, observation, action, reward, done, info, next_observation):
    return reward / 500.0

reward_handler = RewardScaler()
```

We can now precise the `reward_handler` in the `fit` method.

```python
playground.fit(500, verbose=1, reward_handler=reward_handler)
```

Now, our agent performs better than before ! But is there a way to evaluate that without looking at some demo run ? Of course, let's log a few data to compare !

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-standard_agent_normalization-py)

## Logging

### Tensorboard

We are going to use the Tensorboard callback provided by LearnRL to log some data during training.

Let's define two callbacks with `log_dir`, `run_name` and the [metrics](https://learnrl.readthedocs.io/en/latest/callbacks.html#metric-codes) to log.

```python
from learnrl.callbacks import TensorboardCallback

tensorboard_without_normalization = TensorboardCallback(
  log_dir='logs', run_name='without_reward_normalization',
  step_metrics=['reward', 'loss'],
  episode_metrics=['reward.sum', 'loss'],
  cycle_metrics=['reward', 'loss']
)

tensorboard_with_normalization = TensorboardCallback(
  log_dir='logs', run_name='with_reward_normalization',
  step_metrics=['reward', 'loss'],
  episode_metrics=['reward.sum', 'loss'],
  cycle_metrics=['reward', 'loss']
)
```

Let's get some data by running two agents, one with reward normalization and the other without.

```python
custom_action_value1 = MyEstimator(observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   learning_rate=1e-2)
agent1 = StandardAgent(observation_space=env.observation_space,
                       action_space=env.action_space,
                       action_values=custom_action_value1)

playground = rl.Playground(env, agent1)
playground.fit(500, verbose=1, callbacks=[tensorboard_without_normalization])

custom_action_value2 = MyEstimator(observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   learning_rate=1e-2)
agent2 = StandardAgent(observation_space=env.observation_space,
                       action_space=env.action_space,
                       action_values=custom_action_value2)

playground = rl.Playground(env, agent2)
playground.fit(500, verbose=1, reward_handler=reward_handler, callbacks=[tensorboard_with_normalization])
```

We do both agents in one run here, but logs are saved between runs. So, it is also possible to run the program two times : one with normalization and the other one without.

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-standard_agent_logging-py)

You can see the results with the following command in your browser.

```bash
tensorboard --logdir logs
```

The metric `'reward'` corresponds to the reward returned by `env.step(...)`, that's why its value is not normalized yet. The normalized one is `'reward_handled'`.

### Wandb

Another logging tool has a built-in support in LearnRL, it is [Weights and Biaises](https://www.wandb.com/). In order to use it, you need a wandb account and to create a project for this example. Once that is done, you just have to add the following lines to your file :

```python
from learnrl.callbacks import WandbCallback
import wandb
wandb.init(project='your-project-name')

...

playground.fit(200, verbose=1, callbacks=[WandbCallback()])
```

Here, replace `'your-project-name'` by the name of your project on wandb. You can now run your script and you will see your runs on your project dashboard.

You can therefore use wandb sweep to do hyperparameters optimization now!

[Full code](https://gist.github.com/Cr4zySheep/c0b0a9d079feaaca1cf0fa228450268f#file-standard_agent_wandb-py)

## Tweaking the parameters

You are now ready to define your own agents and to log their performance in order to compare them. We can now introduce some parameters that already exists in LearnRL.

Replay buffer is already included inside LearnRL. When creating an agent with `StandardAgent`, the argument `sample_size` define the maximum number of samples to learn from and `forget_after_update` when `True` deactivates the replay buffer.

Another tool to stabilize the learning is to freeze the estimator during a short period to get indepedant sample to train from. This can be done with the argument `freezed_steps` of `KerasEstimator`.

You also have the ability to control the exploration policy of the agent, either by giving your own `Control` object or by giving the arguments `exploration` and `exploration_decay` (The default `Control` is an epsilon-greedy one) to `StandardAgent`.

Play with the different parameters in order to find what works best ! Don't forget to give your different sets of parameters a different `run_name` inside the `TensorboardCallback` to be able to compare them! 
