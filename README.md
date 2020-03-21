# LearnRL
A library to learn and test reinforcement learning algorithms

# Soon Release 1 will be up !

## General framework for reinforcement learning agents

Every agents can be implemented as a subclass of this general agent !
QLearning, DeepQLearning, A3C based agents ... and this librairie tries to prove it !

### Memory

All the agents share the same memory system optimized with numpy to speedup computation
See agents.agent for more details

### Table Agent (With any controls or evaluations) including the famous QLearning !

Table agents builds a table of actions values Q(state, action)
From this table the agent uses a Control algorithm to decide what action to play in each state:
    Control: state(s) -> action to play

Throught experience it updates this table using an evaluation algorithm:
    Evaluation: experience -> Q update

The QLearningAgent is already pre-built and ready for use ! Just import it from agents !

See agents.basic.agent for more details

#### Control

Commons controls are built-in : Greedy, UpperConfidenceBound(UCB), PolynomialUpperConfidenceTrees(Puct)
But you can build any control algorithm by extanding the class Control !
You just have to code the function policy:
    policy(state(s), action_values(Q), action_visits(N)) -> probability of each action

See agents.basic.control for more details

#### Evaluation

Commons evaluations are built-in : MonteCarlo, TemporalDifference(lambda)
But you can build any control algorithm by extanding the class Evaluation !
You just have to code the function learn:
    learn(action_values(Q), action_visits(N), memory:Memory) -> update Q and N

See agents.basic.evaluation for more details

