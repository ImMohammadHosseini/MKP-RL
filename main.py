
"""

"""
import optparse
import pickle

import torch
import numpy as np
from numpy import genfromtxt
from os import path, makedirs

from tqdm import tqdm
#from RL.src.core import LengthSampler
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from configs.ppo_configs import PPOConfig
from configs.fraction_sac_configs import FractionSACConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.EncoderMLP import EncoderMLPKnapsack, RNNMLPKnapsack
from src.models.critic_model import CriticNetwork1, CriticNetwork2
from src.data_structure.state_prepare import StatePrepare
from src.transformer_trainer import TransformerTrainer
from solve_algorithms.RL.src.env import KnapsackAssignmentEnv
from solve_algorithms.random_select import RandomSelect
from solve_algorithms.greedy_select import GreedySelect
import matplotlib.pyplot as plt

from solve_algorithms.RL.whole_ppo_trainer import (
    WholePPOTrainer_t1,
    WholePPOTrainer_t2,
    WholePPOTrainer_t3
    )
from solve_algorithms.RL.fraction_ppo_trainer import (
    FractionPPOTrainer_t1,
    FractionPPOTrainer_t2,
    FractionPPOTrainer_t3
    )
from solve_algorithms.RL.fraction_sac_trainer import FractionSACTrainer
from solve_algorithms.RL.encoder_ppo_trainer import (
    EncoderPPOTrainer_t2, 
    EncoderPPOTrainer_t3
    )
from solve_algorithms.RL.encoder_mlp_sac_trainer import EncoderMlpSACTrainer
usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dim", default=5)
parser.add_option("-O", "--objective", action="store", dest="obj", default=5)
parser.add_option("-K", "--knapsaks", action="store", dest="kps", default=2)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=15)
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
MAIN_BATCH_SIZE = 1
NO_CHANGE_LONG = 5#*BATCH_SIZE#int(1/8*(opts.instances // INSTANCE_OBS_SIZE))
PROBLEMS_NUM = 100#1*MAIN_BATCH_SIZE
N_TRAIN_STEPS = 100000
#NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
DATA_PATH = 'dataset/dim_'+str(opts.dim)+'_obj_'+str(opts.obj)+'/'+opts.mode

PPO_ALGORITHMS = ['EncoderPPOTrainer', 'FractionPPOTrainer', 'WholePPOTrainer']
SAC_ALGORITHMS = ['SACTrainer', 'FractionSACTrainer']
OTHER_ALGORITHMS = ['RandomSelect', 'GreedySelect']
INPUT_TYPES = ['type_1']
OUTPUT_TYPES = ['type_1', 'type_2', 'type_3']
ACTOR_MODEL_TYPES = ['Encoder_MLP', 'Transformer']
CRITIC_MODEL_TYPES = ['MLP']
TRAIN_PROCESS = ['normal_train', 'extra_train']

def dataInitializer ():
    statePrepareList = []
    if path.exists(DATA_PATH):
        instances_weights = [genfromtxt(DATA_PATH+'weights/'+str(i)+'.csv', delimiter=',')
                              for i in range(3)]
        instances_values = [genfromtxt(DATA_PATH+'values/'+str(i)+'.csv', delimiter=',')
                              for i in range(3)]
        ks_caps = [genfromtxt(DATA_PATH+'caps/'+str(i)+'.csv', delimiter=',')
                              for i in range(3)]
        
    else:
        instances_weights, instances_values, ks_caps = multiObjectiveDimentional(
            opts.dim, opts.kps, opts.instances, INFOS)
        
    if not path.exists(DATA_PATH):
        makedirs(DATA_PATH)
        for i,w in enumerate(instances_weights):
            np.savetxt(DATA_PATH+'weights/'+str(i)+'.csv', w, delimiter=",")
        for i,v in enumerate(instances_values):
            np.savetxt(DATA_PATH+'values/'+str(i)+'.csv', v, delimiter=",")
        for i,c in enumerate(ks_caps):
            np.savetxt(DATA_PATH+'caps/'+str(i)+'.csv', c, delimiter=",")
            
    return ks_caps, instances_weights, instances_values

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

def ppoInitializer (output_type, algorithm_type):
    ppoConfig = PPOConfig(generat_link_number=1) if algorithm_type == 'EncoderPPOTrainer' else PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim, DEVICE, ppoConfig.generat_link_number,opts.dim+1)
    
    env = KnapsackAssignmentEnv(modelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE,device=DEVICE)
    statePrepare = StatePrepare(INFOS)
    env.setStatePrepare(statePrepare)
    
    if algorithm_type == 'EncoderPPOTrainer':
        actorModel = EncoderMLPKnapsack(modelConfig, output_type, device=DEVICE)
        normalCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          device=DEVICE, name='extraCriticModel')
        
        if output_type == 'type2':
            ppoTrainer = EncoderPPOTrainer_t2(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                              actorModel, normalCriticModel, extraCriticModel)
            flags = [True, False]

        elif output_type == 'type3': 
            ppoTrainer = EncoderPPOTrainer_t3(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                              actorModel, normalCriticModel, extraCriticModel)
            flags = [True, False]
        
    elif algorithm_type == 'FractionPPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, output_type, device=DEVICE)
        normalCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                          device=DEVICE, name='extraCriticModel')
        
        if output_type == 'type1': 
            ppoTrainer = FractionPPOTrainer_t1(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
            
        elif output_type == 'type2':
            ppoTrainer = FractionPPOTrainer_t2(INFOS, SAVE_PATH, ppoConfig, opts.dim,
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
            
        elif output_type == 'type3':
            ppoTrainer = FractionPPOTrainer_t3(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
        
    elif algorithm_type == 'WholePPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, output_type, device=DEVICE)
        normalCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                          device=DEVICE, name='extraCriticModel')  
        if output_type == 'type1': 
            ppoTrainer = WholePPOTrainer_t1(INFOS, SAVE_PATH, ppoConfig, opts.dim,
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]
            
        elif output_type == 'type2': 
            ppoTrainer = WholePPOTrainer_t2(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]
            
        elif output_type == 'type3':
            ppoTrainer = WholePPOTrainer_t3(INFOS, SAVE_PATH, ppoConfig, opts.dim, 
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]

    return env, ppoTrainer, flags


def ppo_train_extra (env, ppoTrainer, flags, c, w, v, output_type, algorithm_type):
    #statePrepares = np.array(statePrepareList)
    
    #greedyScores = np.array(greedyScores)
    best_reward = -1e4
    reward_history = []; score_history = []; remain_cap_history = []
    n_steps = 0
    n_steps1 = 0
    addition_steps = ppoTrainer.config.generat_link_number if flags[0] else 1
    for i in tqdm(range(N_TRAIN_STEPS)):
        #for batch in range(len(statePrepares)):
        pn = np.random.randint(PROBLEMS_NUM)
        env.statePrepare.setProblem(c[pn], w[pn], v[pn])

        externalObservation, _ = env.reset()
        done = False
        episodeNormalReward = torch.tensor (0.0)
        while not done:
            internalObservations, actions, accepted_action, probs, values, \
                rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                       env.statePrepares, flags[0], flags[1])
            #print(torch.cat(rewards,0).sum())
            episodeNormalReward += torch.cat(rewards,0).sum()
            externalObservation_, extraReward, done, info = env.step(accepted_action)
            
            if episodeNormalReward < -10: done = True
            ppoTrainer.memory.save_normal_step(externalObservation, actions, \
                                               probs, values, rewards, done, \
                                               steps, internalObservations)
            
            n_steps += addition_steps
            if n_steps % ppoTrainer.config.normal_batch == 0:
                ppoTrainer.train('normal')
            '''if ~(accepted_action == [-1,-1]).all():
                #print(accepted_action)
                n_steps1 += addition_steps
                ppoTrainer.memory.save_extra_step(externalObservation, actions, \
                                                  probs, values, [extraReward], \
                                                  done, steps, internalObservations)   
                if n_steps1 % ppoTrainer.config.extra_batch == 0:
                    ppoTrainer.train('extra')'''
            externalObservation = externalObservation_
            
        scores, remain_cap_ratios = env.final_score()
        #batch_score_per_grredy = scores/greedyScores[batch]
        
        reward_history.append(float(episodeNormalReward))
        score_history.append(scores)#batch_score_per_grredy)
        remain_cap_history.append(remain_cap_ratios)
        avg_reward = np.mean(reward_history[-50:])
        avg_score = np.mean(score_history[-50:])
            
        if avg_reward > best_reward:
            best_reward  = avg_reward
            ppoTrainer.save_models()
            ooo
        print('episode', i, 'score %.3f' % scores, 'avg score %.2f' % avg_score,
              'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
              'interanl_reward %.3f'%float(episodeNormalReward), 'avg reward %.3f' %avg_reward)
    
        if i % 1000 == 0:
            results_dict = {'reward': reward_history, 'score': score_history, 'remain_cap': remain_cap_history}
            with open('train_results/'+algorithm_type+'_'+output_type+'_dim'+str(opts.dim)+'_obj_'+str(opts.obj)+'.pickle', 'wb') as file:
                pickle.dump(results_dict, file)
            
    
    
    
    

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
    
def ppo_test(env, ppoTrainer, flags, statePrepareList, greedyScores,
                     output_type, algorithm_type):
    statePrepares = np.array(statePrepareList)
    
    greedyScores = np.array(greedyScores)
    #best_reward = -1e4
    #reward_history = []; score_history = []; remain_cap_history = []
    #n_steps = 0
    #n_steps1 = 0
    #addition_steps = ppoTrainer.config.generat_link_number if flags[0] else 1
    #for i in tqdm(range(N_TRAIN_STEPS)):
    for batch in range(len(statePrepares)):
        env.setStatePrepare(statePrepares[0])

        externalObservation, _ = env.reset()
        done = False
        #episodeNormalReward = torch.tensor (0.0)
        while not done:
            internalObservations, actions, accepted_action, probs, values, \
                rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                       env.statePrepares, flags[0], flags[1])
            #print(torch.cat(rewards,0).sum())
            #episodeNormalReward += torch.cat(rewards,0).sum()
            externalObservation_, extraReward, done, info = env.step(accepted_action)
            
            #if episodeNormalReward < -10: done = True
            #ppoTrainer.memory.save_normal_step(externalObservation, actions, \
            #                                   probs, values, rewards, done, \
            #                                   steps, internalObservations)
            
            #n_steps += addition_steps
            #if n_steps % ppoTrainer.config.normal_batch == 0:
            #    ppoTrainer.train('normal')
            '''if ~(accepted_action == [-1,-1]).all():
                #print(accepted_action)
                n_steps1 += addition_steps
                ppoTrainer.memory.save_extra_step(externalObservation, actions, \
                                                  probs, values, [extraReward], \
                                                  done, steps, internalObservations)   
                if n_steps1 % ppoTrainer.config.extra_batch == 0:
                    ppoTrainer.train('extra')'''
            externalObservation = externalObservation_
            
        scores, remain_cap_ratios = env.final_score()
        batch_score_per_grredy = scores/greedyScores[batch]
        
        #reward_history.append(float(episodeNormalReward))
        #score_history.append(batch_score_per_grredy)
        #remain_cap_history.append(remain_cap_ratios)
        #avg_reward = np.mean(reward_history[-50:])
        #avg_score = np.mean(score_history[-50:])
        
        #if avg_reward > best_reward:
        #    best_reward  = avg_reward
        #    ppoTrainer.save_models()
        print('score %.3f' % batch_score_per_grredy, 
              'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios))

if __name__ == '__main__':
    
    c, w, v = dataInitializer()
    #greedyScores = greedyAlgorithm(statePrepareList)
    
    #for algorithm_type in PPO_ALGORITHMS:
    #    for output_type in OUTPUT_TYPES:False, False
    env, ppoTrainer, flags = ppoInitializer ('type3', 'EncoderPPOTrainer')
    if opts.mode == 'train':
        ppo_train_extra(env, ppoTrainer, flags, 'type3', 'EncoderPPOTrainer')
    elif opts.mode == 'test':
        ppo_test(env, ppoTrainer, flags, 'type3', 'EncoderPPOTrainer')
    