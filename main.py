
"""

"""
import optparse
import torch
import numpy as np
from numpy import genfromtxt
from os import path, makedirs

from tqdm import tqdm
#from RL.src.core import LengthSampler
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from configs.ppo_configs import PPOConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.EncoderMLP import EncoderMLPKnapsack
from src.models.critic_model import CriticNetwork, ExternalCriticNetwork
from src.data_structure.state_prepare import ExternalStatePrepare
from src.transformer_trainer import TransformerTrainer
from solve_algorithms.RL.src.env import KnapsackAssignmentEnv
from solve_algorithms.random_select import RandomSelect
from solve_algorithms.greedy_select import GreedySelect
import matplotlib.pyplot as plt

from solve_algorithms.RL.ppo_trainer import PPOTrainer
from solve_algorithms.RL.fraction_ppo_trainer import FractionPPOTrainer

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dim", default=5)
parser.add_option("-K", "--knapsaks", action="store", dest="kps", default=5)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=20)
parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':80,
         'CAP_HIGH':350, 
         'WEIGHT_LOW':10, 
         'WEIGHT_HIGH':100,
         'VALUE_LOW':3, 
         'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = opts.instances#2 * KNAPSACK_OBS_SIZE
BATCH_SIZE = 1
NO_CHANGE_LONG = 1#*BATCH_SIZE#int(1/8*(opts.instances // INSTANCE_OBS_SIZE))
PROBLEMS_NUM = 1*BATCH_SIZE
N_TRAIN_STEPS = 200000
#NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
MAIN_DATA = 'dataset/'
ALGORITHMS_NAME = ['PPOTrainer', 'FractionPPOTrainer' 'A2C', 
                   'RandomSelect', 'GreedySelect']

def dataInitializer ():
    statePrepareList = []
    if path.exists(MAIN_DATA):
        instance_main_data = genfromtxt(MAIN_DATA+'instances.csv', delimiter=',')
        ks_main_data = genfromtxt(MAIN_DATA+'ks.csv', delimiter=',')
    for _ in range(PROBLEMS_NUM):
        if opts.var == 'multipleKnapsack':
            c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
        elif opts.var == 'multipleKnapsack':
            pass
        elif opts.var == 'multiObjectiveDimentional':
            c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
        
        if not path.exists(MAIN_DATA):
            instance_main_data = w
            ks_main_data = c
            makedirs(MAIN_DATA)
            np.savetxt(MAIN_DATA+'instances.csv', instance_main_data, delimiter=",")
            np.savetxt(MAIN_DATA+'ks.csv', ks_main_data, delimiter=",")

        statePrepare = ExternalStatePrepare(c, w, v, ks_main_data, instance_main_data, 
                                            KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE)
        
        #statePrepare.normalizeData(INFOS['CAP_HIGH'], INFOS['VALUE_HIGH'])
        statePrepareList.append(statePrepare)            
    return statePrepareList

def randomAlgorithm (statePrepare):
    test_score_history = []
    #for i in tqdm(range(N_TEST_STEPS)):
    random_select = RandomSelect (statePrepare)
    random_select.test_step()#TODO
    
def greedyAlgorithm (statePrepareList):
    greedyScores = []; 
    for statePrepare in statePrepareList:
        greedy_select = GreedySelect(opts.instances, statePrepare)
        bestScore = 0
        for _ in range(10):
            score, _ = greedy_select.test_step()
            if score > bestScore:
                bestScore = score
        greedyScores.append(bestScore)
    return greedyScores

def fractionPPOInitializer ():
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    actorModel = TransformerKnapsack(modelConfig, ppoConfig.generat_link_number, DEVICE)#EncoderMLPKnapsack
    criticModel = CriticNetwork(modelConfig.max_length*modelConfig.input_encode_dim,
                                (ppoConfig.generat_link_number+1)*modelConfig.input_decode_dim)
    externalCriticModel = ExternalCriticNetwork(modelConfig.max_length, modelConfig.input_encode_dim)
    env = KnapsackAssignmentEnv(modelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, BATCH_SIZE, DEVICE)
    
    ppoTrainer = FractionPPOTrainer(INFOS, SAVE_PATH, BATCH_SIZE, ppoConfig, actorModel, 
                                    criticModel, externalCriticModel, DEVICE)
    return env, ppoTrainer

def ppoInitializer ():
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    actorModel = TransformerKnapsack(modelConfig, ppoConfig.generat_link_number, DEVICE)#EncoderMLPKnapsack

    criticModel = ExternalCriticNetwork(modelConfig.max_length, modelConfig.input_encode_dim)
    
    env = KnapsackAssignmentEnv(modelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, BATCH_SIZE, DEVICE)
    ppoTrainer = PPOTrainer(INFOS, SAVE_PATH, BATCH_SIZE, ppoConfig, actorModel, 
                            criticModel, DEVICE)
    
    return env, ppoTrainer
    
    
def plot_learning_curve(x, scores, figure_file, title):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.title(title)
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



def fraction_ppo_train (env, ppoTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    if ppoTrainer.pretrain_need:
        transformer_trainer = TransformerTrainer(ppoTrainer.config.generat_link_number,
                                                 statePrepareList, ppoTrainer.actor_model, 
                                                 device=DEVICE)
        transformer_trainer.train(opts.instances, BATCH_SIZE)
    greedyScores = np.array(greedyScores)
    best_score = .9
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, BATCH_SIZE)
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])

            externalObservation, _ = env.reset()
            done = False
            while not done:
                internalObservations, actions, accepted_actions, probs, values, \
                    internalRewards, steps = ppoTrainer.make_steps(
                        externalObservation, env.statePrepares, done)
                externalObservation_, externalReward, done, info = env.step(accepted_actions)
                ppoTrainer.save_step (externalObservation, internalObservations,
                                      actions, probs, values, internalRewards,
                                      steps, done)
                n_steps += ppoTrainer.config.generat_link_number
                if n_steps % ppoTrainer.config.internal_batch == 0:
                    ppoTrainer.train_minibatch()
                externalObservation = externalObservation_
                
            scores, remain_cap_ratios = env.final_score()
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])

            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score:
                best_score = avg_score
                ppoTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios))
    
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title)
    
def ppo_train (env, ppoTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    if ppoTrainer.pretrain_need:
        transformer_trainer = TransformerTrainer(ppoTrainer.config.generat_link_number,
                                                 statePrepareList, ppoTrainer.actor_model, 
                                                 device=DEVICE)
        transformer_trainer.train(opts.instances, BATCH_SIZE)
    greedyScores = np.array(greedyScores)
    best_score = .65
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, BATCH_SIZE)
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])

            externalObservation, _ = env.reset()
            done = False
            while not done:
                internalObservatio, actions, accepted_acctions, sumProbs, sumVals, \
                    sumRewards, steps = ppoTrainer.make_steps(
                    externalObservation, env.statePrepares, done)
                externalObservation_, externalReward, done, info = env.step(accepted_acctions)
                ppoTrainer.save_step (externalObservation, internalObservatio,
                                      actions, sumProbs, sumVals, sumRewards,
                                      steps, done)
                n_steps += 1
                if n_steps % ppoTrainer.config.internal_batch == 0:
                    ppoTrainer.train_minibatch()
                externalObservation = externalObservation_
            
            scores, remain_cap_ratios = env.final_score()
            
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])

            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score:
                best_score = avg_score
                ppoTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios))
    
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title)
    

if __name__ == '__main__':
    
    statePrepareList = dataInitializer()
    greedyScores = greedyAlgorithm(statePrepareList)
    

    env, ppoTrainer = fractionPPOInitializer()

    if opts.mode == "train" :
        fraction_ppo_train(env, ppoTrainer, statePrepareList, greedyScores)
    else:
        rl_test(env, ppoTrainer, statePrepareList)#TODO waaayyy
