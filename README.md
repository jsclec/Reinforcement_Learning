# Reinforcement_Learning


Here I have compiled some reinforcement learning projects utilizing different approaches......
The testbed for all of these algorithms and approaches is OpenAI's gym environment. It is open source and available at https://github.com/openai/gym .For simplicity the agents used in these programs learn to play CartPole and MountainCar, both seminal games in classic control.


Credit where credit is due: All of these algorithms were adapted from the de facto RL textbook, Reinforcement Learning: an Introduction by Sutton and Barto, and the blueprint for these programs extended from the course Reinforcement Learning at UT Austin taught by Profs. Peter Stone and Scott Niekum. If you are a student, please do not try and copy these for an assignment, it robs you of the opportunity to learn and defeats the purpose of getting an education. I have intentionally left out the necessary testing files and policy files to prevent this (or at least make it difficult).


Playing Mountain Car with on-policy n_step td and off-policy n-step sarsa:
* Given a trajectory, find the optimal policy or a reasonably accurate value function for the current policy

Off-Policy Monte Carlo with Ordinary importance sampling and weighted importance sampling:
* Given a simulated trajectory, return the action-state pair value function Q

Value prediction and iteration with Dynamic Programming:
* Use DP to iteratively update value functions until they converge to a reasonable uncertainty

REINFORCE with and without baseline:
* 

SARSA with neural network function approximation:
* 
