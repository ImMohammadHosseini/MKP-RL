# :zap:MKP-RL:zap:

[![-----------------------------------------------------]( 
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/ImMohammadHosseini/incremental-learning?tab=repositories)

## :bookmark: 1- Introduction

In this GitHub repository, we present a new approach to solving the multi-dimensional multiple knapsack problem. This work stems from my master's thesis focused on resource management in cloud computing. To investigate and test my thesis, I initially conducted experiments using a simplified environment, prior to considering the complexities inherent in cloud computing resource management. The primary objective of our project is to train a transformer model using reinforcement learning algorithms, such as Proximal Policy Optimization (PPO). We have implemented two variants of PPO for this purpose.

The ultimate goal of this project is to leverage the trained transformer model to effectively highlight the object-knapsack connections during various stages of the problem-solving process. By addressing the multi-dimensional multiple knapsack problem, we aim to contribute to the advancement of cloud computing resource management techniques.

The knapsack problem refers to a classic optimization problem where a set of objects, each characterized by a weight and value, must be packed into a limited-capacity knapsack. The objective is to maximize the total value of the chosen objects while adhering to the constraint of not exceeding the knapsack's weight limit. Intriguingly, the knapsack problem exhibits similarities to resource management in cloud computing, where cloud resources must be efficiently allocated to optimize performance and cost.

While this project demonstrates great potential, :bug: there is an issue or bug that needs to be addressed. As specified in the issues section, the current problem lies in the model's training, which does not yield satisfactory performance. By sharing the issue details, I hope to collaborate with others and collectively find a solution. :rocket:

## :bookmark: 2- Proposed Method
### 1-2- PPO Background
Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions and uses this feedback to improve its decision-making abilities over time. The goal is to find the optimal policy that maximizes the cumulative rewards obtained from the environment.

Proximal Policy Optimization (PPO) is a popular algorithm used in reinforcement learning. PPO is designed to optimize the policy, which is the strategy used by the agent to make decisions. It is an on-policy method, meaning it collects data by interacting with the environment using the current policy.

PPO uses a surrogate objective function to estimate the loss between the new policy and the old policy. It then applies optimization techniques to update the policy in a way that maximizes the expected rewards while ensuring that the policy update is not too large. By gradually improving the policy through multiple iterations, PPO is able to find a locally optimal policy.

PPO has gained attention due to its ability to strike a balance between sample efficiency and stability. It avoids the need for value function approximation and provides an easy-to-use algorithm for reinforcement learning tasks.

### 2-2- RL System Model
**STATE:** In our system model, each state is perceived as an external observation represented by a 3D matrix. The first dimension comprises all information related to one problem (our methodology allows for the inclusion of multiple problems). The second dimension accommodates the maximum input size of the transformer model, which is equal to the sum of the instanceobservationsize, knapsackobservationsize, and 3. Lastly, the third dimension contains information regarding instances and knapsacks, incorporating all n-dimensional instances along with an additional dimension for value, followed by all knapsack n-dimensions and an extra dimension filled with zeroes.

Within the third dimension of the state, we introduce three tokens: SOD (start of data), EOD (end of data), and PAD. SOD is represented by an n+1-dimensional vector with all dimensions filled with 1, serving as the start token. Conversely, EOD is an n+1-dimensional vector with all dimensions set to 2, functioning as the end token and separating the instance and knapsack data. In cases where the number of remaining instances in a state is less than instanceobservationsize, we incorporate PAD tokens (n+1-dimensional vector with all dimensions set to 0) before the SOD token.

![The structure](images/fig1.jpg)

**ACTION:**
**REWARD-FUNCTION:**
## :bookmark: 3-Functionality and Methodology

### 1-3-Data Format and Preprocessing part

### 2-3-Transformer as Actor Model
#### 1-2-3-Encoder-Decoder Model
#### 2-2-3-Encoder-Mlp Model

### 3-3-RL Algorithms
#### 1-3-3-PPO
#### 2-3-3-Fraction-PPO
