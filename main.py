
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
                  default=50)
parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='train')
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':20,#150, 
         'CAP_HIGH':200, 
         'WEIGHT_LOW':10, 
         'WEIGHT_HIGH':100,
         'VALUE_LOW':3, 
         'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = 50#2 * KNAPSACK_OBS_SIZE
BATCH_SIZE = 4
NO_CHANGE_LONG = 1#*BATCH_SIZE#int(1/8*(opts.instances // INSTANCE_OBS_SIZE))
PROBLEMS_NUM = 2*BATCH_SIZE
N_TRAIN_STEPS = 10000
#NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
ALGORITHMS_NAME = ['PPOTrainer', 'A2C', 'RandomSelect', 'GreedySelect']

def dataInitializer ():
    statePrepareList = []
    for _ in range(PROBLEMS_NUM):
        if opts.var == 'multipleKnapsack':
            c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
        elif opts.var == 'multipleKnapsack':
            pass
        elif opts.var == 'multiObjectiveDimentional':
            c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
        
        statePrepare = ExternalStatePrepare(c, w, v, KNAPSACK_OBS_SIZE,
                                            INSTANCE_OBS_SIZE)
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
        random_select = GreedySelect(statePrepare)
        bestScore = 0
        for _ in range(5):
            score = random_select.test_step()
            if score > bestScore:
                bestScore = score
        greedyScores.append(bestScore)
    return greedyScores

   
def rlInitializer ():
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    actorModel = TransformerKnapsack(modelConfig, DEVICE)
    ref_model = TransformerKnapsack(modelConfig, DEVICE)

    criticModel = CriticNetwork(modelConfig.max_length*modelConfig.input_dim,
                                (ppoConfig.generat_link_number)*12)#modelConfig.output_dim)
    externalCriticModel = ExternalCriticNetwork(modelConfig.max_length*modelConfig.input_dim)
    env = KnapsackAssignmentEnv(modelConfig.input_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, BATCH_SIZE, DEVICE)
    ppoTrainer = PPOTrainer(INFOS, SAVE_PATH, BATCH_SIZE, ppoConfig, actorModel, 
                            ref_model, criticModel, externalCriticModel, DEVICE)
    
    return env, ppoTrainer

def plot_learning_curve(x, scores, figure_file, title):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
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

def rl_train (env, ppoTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    #for statePrepare in statePrepares :
    #    statePrepare.normalizeData(INFOS['CAP_HIGH'], INFOS['WEIGHT_HIGH'], INFOS['VALUE_HIGH'])

    greedyScores = np.array(greedyScores)
    best_score = .8
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        _, batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, BATCH_SIZE)
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])

            externalObservation, _ = env.reset()
            done = False
            while not done:
                step_acts, step_prob = ppoTrainer.steps(externalObservation, 
                                                        env.statePrepares,
                                                        done)
                #externalObservations = torch.cat([externalObservations]+[externalObservation]*len(step_acts), 0)
                externalObservation, externalReward, done, info = env.step(step_acts, statePrepares[batch[0]])
                #externalRewards = torch.cat([externalRewards, torch.tensor(externalReward, device=DEVICE)], 0)
                #accepted_act = torch.cat([accepted_act, torch.tensor(step_acts, device=DEVICE)], 0)
                #accepted_prob = torch.cat([accepted_prob, torch.tensor(step_prob, device=DEVICE)])
                n_steps += 1
            #ppoTrainer.external_train(externalObservations, accepted_act, accepted_prob,
            #                          torch.tensor(externalRewards, device=DEVICE), 
            #                          env.statePrepare)
            scores, remain_cap_ratios = env.final_score()
            #print(scores)
            #print(greedyScores[batch])
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])

            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_score = np.mean(score_history[-10:])

            if avg_score > best_score:
                best_score = avg_score
                ppoTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio', np.mean(remain_cap_ratios))
    
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/score_per_greedyScore.png'
    title = 'Running average of previous 10 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/remain_cap_ratio.png'
    title = 'Running average of previous 10 remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title)
    

if __name__ == '__main__':
    
    statePrepareList = dataInitializer()
    greedyScores = greedyAlgorithm(statePrepareList)
    env, ppoTrainer = rlInitializer()

    if opts.mode == "train" :
        rl_train(env, ppoTrainer, statePrepareList, greedyScores)
    else:
        rl_test(env, ppoTrainer, statePrepareList)#TODO waaayyy
