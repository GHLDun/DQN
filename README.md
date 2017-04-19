# DQN
by chenzhaomin

### Overview
This repository is an implementation of  the [dqn](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) method which  is used to solve the [OpenAI CartPole-v0](https://gym.openai.com/evaluations/eval_NZKl9J8wTHC3VNQNREUt2Q), and the reinforcement_q_learning.py is the file from [pytorch tutorials](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

## Installation Dependencies:
* Python2.7
* Numpy
* Pytorch

## How to Run?
```
git clone https://github.com/chenzhaomin123/DQN.git
cd DQN
python CartPole-v0.py
```

## Deep Q-Network Algorithm

The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```