
"""

"""
import optparse
import torch
import numpy as np
from os import path
from tqdm import tqdm
#from RL.src.core import LengthSampler
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from configs.ppo_configs import PPOConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.critic_model import CriticNetwork, ExternalCriticNetwork
from src.data_structure.state_prepare import ExternalStatePrepare
from solve_algorithms.RL.src.env import KnapsackAssignmentEnv
from solve_algorithms.random_select import RandomSelect
from solve_algorithms.greedy_select import GreedySelect
import matplotlib.pyplot as plt

from solve_algorithms.RL.ppo_trainer import PPOTrainer

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dim", default=5)
parser.add_option("-K", "--knapsaks", action="store", dest="kps", default=10)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=100)
parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':40,#150, 
         'CAP_HIGH':400, 
         'WEIGHT_LOW':10, 
         'WEIGHT_HIGH':100,
         'VALUE_LOW':3, 
         'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = 100#2 * KNAPSACK_OBS_SIZE
NO_CHANGE_LONG = 5#int(1/8*(opts.instances // INSTANCE_OBS_SIZE))
N_TRAIN_STEPS = 300
NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
ALGORITHMS_NAME = ['PPOTrainer', 'A2C', 'RandomSelect', 'GreedySelect']

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

def randomAlgorithm (statePrepare):
    test_score_history = []
    #for i in tqdm(range(N_TEST_STEPS)):
    random_select = RandomSelect (statePrepare)
    random_select.test_step()#TODO
    
def greedyAlgorithm (statePrepare):
    test_score_history = []
    #for i in tqdm(range(N_TEST_STEPS)):
    random_select = GreedySelect (statePrepare)
    return random_select.test_step()#TODO

   
def rlInitializer (statePrepare):
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    actorModel = TransformerKnapsack(modelConfig, DEVICE)
    
    criticModel = CriticNetwork(modelConfig.max_length*modelConfig.input_dim,
                                (2*ppoConfig.generat_link_number)*modelConfig.input_dim)#
    externalCriticModel = ExternalCriticNetwork(modelConfig.max_length*modelConfig.input_dim)
    env = KnapsackAssignmentEnv(modelConfig.input_dim, INFOS, statePrepare, 
                                NO_CHANGE_LONG, DEVICE)
    ppoTrainer = PPOTrainer(INFOS, SAVE_PATH, ppoConfig, actorModel, 
                            criticModel, externalCriticModel, DEVICE)
    
    return env, ppoTrainer

def plot_learning_curve(x, scores, figure_file):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 10 scores')
    plt.savefig(figure_file)

def rl_test (env, ppoTrainer):
    test_score_history = []
    for i in tqdm(range(N_TEST_STEPS)):
        externalObservation, _ = env.reset()
        done = False
        externalReward = 0
        while not done:
            step_acts = ppoTrainer.test_step(externalObservation, 
                                             env.statePrepare)
            externalObservation, externalReward, done, info = env.step(step_acts)
        test_score_history.append(externalReward)
    avg_score = np.mean(test_score_history)
    print('avg_test_scor:', avg_score)

def rl_train (env, ppoTrainer):
    best_score = 0.9#INFOS['VALUE_LOW']
    score_history = []
    n_steps = 0
    sp = [env.statePrepare]; gr = [greedyAlgorithm(env.statePrepare)]
    for i in range(10):
        statePrepare = dataInitializer()
        sp.append(statePrepare)
        gr.append(greedyAlgorithm(statePrepare))
        
    for i in tqdm(range(N_TRAIN_STEPS)):
        #if i % NEW_PROBLEM_PER_EPISODE == NEW_PROBLEM_PER_EPISODE-1:
        #    statePrepare = dataInitializer()
        #    greedyAlgorithm(statePrepare)
        #    env.set_statePrepare(statePrepare)
        rand_choose = np.random.randint(11)
        env.set_statePrepare(sp[rand_choose])
        externalObservation, _ = env.reset()
        externalObservations = torch.zeros([0, ppoTrainer.actor_model.config.max_length,
                                            ppoTrainer.actor_model.config.input_dim], 
                                           device=DEVICE)
        done = False
        externalReward = 0
        accepted_act = torch.zeros((0,2), device=DEVICE)
        accepted_prob = torch.tensor([], device=DEVICE)
        externalRewards = torch.tensor([], device=DEVICE)
        while not done:
            step_acts, step_prob = ppoTrainer.steps(externalObservation, 
                                                    env.statePrepare,
                                                    done)
            externalObservations = torch.cat([externalObservations]+[externalObservation]*len(step_acts), 0)

            externalObservation, externalReward, done, info = env.step(step_acts)
            externalRewards = torch.cat([externalRewards, torch.tensor(externalReward, device=DEVICE)], 0)
            accepted_act = torch.cat([accepted_act, torch.tensor(step_acts, device=DEVICE)], 0)
            accepted_prob = torch.cat([accepted_prob, torch.tensor(step_prob, device=DEVICE)])
            n_steps += 1
        #ppoTrainer.external_train(externalObservations, accepted_act, accepted_prob,
        #                          torch.tensor(externalRewards, device=DEVICE), 
        #                          env.statePrepare)
        #score += externalReward
        score, remain_cap_ratio = env.final_score()
        score_history.append(np.sum(score/gr[rand_choose]))
        avg_score = np.mean(score_history[-10:])

        if avg_score > best_score:
            best_score = avg_score
            ppoTrainer.save_models()
        print('episode', i, 'score %.3f' % (score/gr[rand_choose]), 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'remain_cap_ratio', remain_cap_ratio)
    x = [i+1 for i in range(len(score_history))]
    
    figure_file = 'plots/ks_problem.png'
    plot_learning_curve(x, score_history, figure_file)#TODO add visualization


if __name__ == '__main__':
    
    statePrepare = dataInitializer()
    randomAlgorithm(statePrepare)
    greedyAlgorithm(statePrepare)
    env, ppoTrainer = rlInitializer(statePrepare)

    if opts.mode == "train" :
        rl_train(env, ppoTrainer)
    else:
        rl_test(env, ppoTrainer)#TODO waaayyy
