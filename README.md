# :zap:MKP-RL:zap:

[![-----------------------------------------------------]( 
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/ImMohammadHosseini/incremental-learning?tab=repositories)

## :bookmark: 1-Introduction

In this GitHub repository, we present a new approach to solving the multi-dimensional multiple knapsack problem. This work stems from my master's thesis focused on resource management in cloud computing. To investigate and test my thesis, I initially conducted experiments using a simplified environment, prior to considering the complexities inherent in cloud computing resource management. The primary objective of our project is to train a transformer model using reinforcement learning algorithms, such as Proximal Policy Optimization (PPO). We have implemented two variants of PPO for this purpose.

The ultimate goal of this project is to leverage the trained transformer model to effectively highlight the object-knapsack connections during various stages of the problem-solving process. By addressing the multi-dimensional multiple knapsack problem, we aim to contribute to the advancement of cloud computing resource management techniques.

The knapsack problem refers to a classic optimization problem where a set of objects, each characterized by a weight and value, must be packed into a limited-capacity knapsack. The objective is to maximize the total value of the chosen objects while adhering to the constraint of not exceeding the knapsack's weight limit. Intriguingly, the knapsack problem exhibits similarities to resource management in cloud computing, where cloud resources must be efficiently allocated to optimize performance and cost.

While this project demonstrates great potential, :bug: there is an issue or bug that needs to be addressed. As specified in the issues section, the current problem lies in the model's training, which does not yield satisfactory performance. By sharing the issue details, I hope to collaborate with others and collectively find a solution. :rocket:

## :bookmark: 2-Functionality and Methodology

### 1-2-Data Format and Preprocessing part

### 2-2-Transformer as Actor Model
#### 1-2-2-Encoder-Decoder Model
#### 2-2-2-Encoder-Mlp Model

### 3-2-RL Algorithms
#### 1-3-2-PPO
#### 2-3-2-Fraction-PPO
