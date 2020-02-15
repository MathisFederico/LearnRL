# LearnRL
A library to learn and test reinforcement learning algorithms

# --- In developpement ---

## TODO

### Git
Contributing pipeline (dev branch, feature branches, ...)

### Environements
Include Gym environements composed of:
- action_space, state_space
- step(action) -> observation, reward, done, info
- reset() -> first_observation
- render() -> None;

Build standard for multi-agents environements
Build "evaluation" methode to evaluate multiple agents on a number of games.

### Agents
Agents must have :
- A policy
- A generalised value fuction (May be split on state (V) and action (Q) values only)
- play(observation) -> action; (How can we have an agent that adapt to multiple and unkowned env?)
- render() -> None; method specified for evey type of agent

Agents types in mind :
#### Standard RL agents (If discreet envs)
##### Evaluation
- Monte-Carlo
- TD($$\lambda$$)

##### Control
- greedy
- $$\epsilon$$-greedy
- UCB
- Puck
- Puck/UCB

##### Standard combinaisons
- SARSA
- Q_learning

#### Standard-DeepRL agents

##### Deep-Qlearning
- Need DataStoring for experience replay

##### Actor-Critic
- Need Tensorflow/Pytorch for custom gradient descent

#### MCTS-based Agents
Need for sampling or true/learned model.

##### MCTS

##### AlphaZero
- Need the model !

##### MuZero
