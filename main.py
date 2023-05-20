
"""

"""
import optparse
import torch
import numpy as np
from RL.src.core import LengthSampler
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from configs.ppo_configs import PPOConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.critic_model import CriticNetwork
from src.data_structure.state_prepare import ExternalStatePrepare
from RL.src.env import KnapsackAssignmentEnv
import matplotlib.pyplot as plt

from RL.ppo_trainer import PPOTrainer

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dim", default=5)
parser.add_option("-M", "--knapsaks", action="store", dest="kps", default=60)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=3000)
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':150, 'CAP_HIGH':400, 'WEIGHT_LOW':10, 'WEIGHT_HIGH':100,
         'VALUE_LOW':1, 'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = 2 * KNAPSACK_OBS_SIZE
NO_CHANGE_LONG = int(3/4*(opts.instances // INSTANCE_OBS_SIZE))
N_TRAIN_STEPS = 300

def dataInitializer ():
    if opts.var == 'multipleKnapsack':
       c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
    elif opts.var == 'multipleKnapsack':
        pass
    elif opts.var == 'multiObjectiveDimentional':
        c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
    
    statePrepare = ExternalStatePrepare(c, w, v, KNAPSACK_OBS_SIZE, 
                                        INSTANCE_OBS_SIZE)
    
    return statePrepare
    
def rlInitializer (statePrepare):
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    model = TransformerKnapsack(modelConfig, DEVICE)
    criticModel = CriticNetwork(modelConfig.max_length*modelConfig.input_dim,
                                (ppoConfig.internal_batch+1)*modelConfig.output_dim)
    env = KnapsackAssignmentEnv(modelConfig.input_dim, INFOS, statePrepare, DEVICE)
    ppoTrainer = PPOTrainer(INFOS, ppoConfig, model, criticModel, DEVICE)
    
    return env, ppoTrainer


def plot_learning_curve(x, scores, figure_file):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    
def train (env, ppoTrainer, outputLengthSampler):
    best_score = INFOS['VALUE_LOW']
    score_history = []
    n_steps = 0
    for i in range(N_TRAIN_STEPS):#TODO CHECK THE EPOCH
        externalObservation, _ = env.reset()
        done = False
        score = 0
        while not done:
            print('hi')
            step_acts, learn_iters = ppoTrainer.steps(externalObservation, env.statePrepare)
            externalObservation, externalReward, done, info = env.step(step_acts)
            n_steps += 1
            score += externalReward
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            #agent.save_models()
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    
    figure_file = 'plots/cartpole.png'
    plot_learning_curve(x, score_history, figure_file)#TODO add visualization


def generate ():
    statePrepare = dataInitializer()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE)#TODO check config file
    model = TransformerKnapsack(modelConfig)
    
    statePrepare.reset()
    stateCaps, stateWeightValues = statePrepare.getObservation()
    ACT = np.zeros((1,6))
    observation = np.append(ACT, np.append(np.append(stateWeightValues, ACT, axis=0), np.append(stateCaps, 
                                                   ACT, axis=0),axis=0),
                                                   axis=0)
    observation = torch.tensor(observation).unsqueeze(dim=0)
    observation = torch.cat([observation, observation], 0)
    observation=observation.to(torch.float32)
    generate, internal_obs = model.generate(observation, 1)
    return generate, internal_obs, observation
    
#generate, internal_obs, observation = generate()
if __name__ == '__main__':
    
    statePrepare = dataInitializer()
    env, ppoTrainer = rlInitializer(statePrepare)
    outputLengthSampler = LengthSampler(KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE)
    train(env, ppoTrainer, outputLengthSampler)