
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
from configs.fraction_sac_configs import FractionSACConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.EncoderMLP import EncoderMLPKnapsack, RNNMLPKnapsack
from src.models.critic_model import CriticNetwork1, CriticNetwork2
from src.data_structure.state_prepare import ExternalStatePrepare
from src.transformer_trainer import TransformerTrainer
from solve_algorithms.RL.src.env import KnapsackAssignmentEnv
from solve_algorithms.random_select import RandomSelect
from solve_algorithms.greedy_select import GreedySelect
import matplotlib.pyplot as plt

from solve_algorithms.RL.ppo_trainer import PPOTrainer
from solve_algorithms.RL.fraction_ppo_trainer import FractionPPOTrainer
from solve_algorithms.RL.one_generate_fraction_ppo_trainer import OneGenerateFractionPPOTrainer
from solve_algorithms.RL.fraction_sac_trainer import FractionSACTrainer
from solve_algorithms.RL.encoder_mlp_ppo_trainer import (
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
parser.add_option("-D", "--dim", action="store", dest="dim", default=1)
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
PROBLEMS_NUM = 1*MAIN_BATCH_SIZE
N_TRAIN_STEPS = 10000
#NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models'
DATA_PATH = 'dataset/'

PPO_ALGORITHMS = ['EncoderPPOTrainer', 'FractionPPOTrainer', 'WholePPOTrainer']
PPO_ALGORITHMS = ['SACTrainer', 'FractionSACTrainer']
OTHER_ALGORITHMS = ['RandomSelect', 'GreedySelect']
INPUT_TYPES = ['type_1']
OUTPUT_TYPES = ['type_1', 'type_2', 'type_3']
ACTOR_MODEL_TYPES = ['Encoder_MLP', 'Transformer']
CRITIC_MODEL_TYPES = ['MLP']
TRAIN_PROCESS = ['normal_train', 'extra_train']

def dataInitializer ():
    statePrepareList = []
    if path.exists(DATA_PATH):
        instance_main_data = genfromtxt(DATA_PATH+'instances.csv', delimiter=',')
        w = instance_main_data[:,:-1]
        v = np.expand_dims(instance_main_data[:,-1],1)
        ks_main_data = genfromtxt(DATA_PATH+'ks.csv', delimiter=',')
        c = ks_main_data[:,:-1]
    else:
        for _ in range(PROBLEMS_NUM):
            if opts.var == 'multipleKnapsack':
                c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
            elif opts.var == 'multipleKnapsack':
                pass
            elif opts.var == 'multiObjectiveDimentional':
                c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
        
    if not path.exists(DATA_PATH):
        instance_main_data = np.append(w,v,1)
        ks_main_data = np.append(c, np.zeros((c.shape[0],1)),1)
        makedirs(DATA_PATH)
        np.savetxt(DATA_PATH+'instances.csv', instance_main_data, delimiter=",")
        np.savetxt(DATA_PATH+'ks.csv', ks_main_data, delimiter=",")

    statePrepare = ExternalStatePrepare(INFOS, c, w, v, ks_main_data[:,:-1], 
                                        instance_main_data[:,:-1], 
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

def ppoInitializer (output_type, algorithm_type):
    ppoConfig = PPOConfig(generat_link_number=1) if algorithm_type == 'EncoderPPOTrainer' else PPOConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim, DEVICE, ppoConfig.generat_link_number)
    
    env = KnapsackAssignmentEnv(modelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, MAIN_BATCH_SIZE)
    
    if algorithm_type == 'EncoderPPOTrainer':
        actorModel = EncoderMLPKnapsack(modelConfig, output_type, device=DEVICE)
        normalCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          device=DEVICE, name='extraCriticModel')
        
        if output_type == 'type2':
            ppoTrainer = EncoderPPOTrainer_t2(INFOS, SAVE_PATH, ppoConfig, actorModel, 
                                              normalCriticModel, extraCriticModel)
        elif output_type == 'type3': 
            'h'
            ppoTrainer = EncoderPPOTrainer_t3(INFOS, SAVE_PATH, ppoConfig, actorModel, 
                                              normalCriticModel, extraCriticModel)
        
    elif algorithm_type == 'FractionPPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, output_type, device=DEVICE)
        ppoTrainer = FractionPPOTrainer(INFOS, SAVE_PATH, MAIN_BATCH_SIZE, ppoConfig, 
                                        actorModel, normalCriticModel, extraCriticModel,
                                        DEVICE)
        
    elif algorithm_type == 'WholePPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, output_type, device=DEVICE)
        ppoTrainer = (INFOS, SAVE_PATH, MAIN_BATCH_SIZE, ppoConfig, 
                                        actorModel, normalCriticModel, extraCriticModel,
                                        DEVICE)
    return env, ppoTrainer

def fractionSACInitializer ():
    sacConfig = FractionSACConfig()
    actorModelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                                 opts.dim, output_dim=12)
    actorModel = TransformerKnapsack(actorModelConfig, sacConfig.generat_link_number, DEVICE)
    
    criticModelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                                 opts.dim, output_dim=32)
    criticLocal1 = TransformerKnapsack(criticModelConfig, sacConfig.generat_link_number, 
                                       device=DEVICE, name = 'criticLocal1', 
                                       out_mode = 'lin_layer')
    criticLocal2 = TransformerKnapsack(criticModelConfig, sacConfig.generat_link_number, 
                                       device=DEVICE, name = 'criticLocal2', 
                                       out_mode = 'lin_layer')
    criticTarget1 = TransformerKnapsack(criticModelConfig, sacConfig.generat_link_number, 
                                       device=DEVICE, name = 'criticTarget1', 
                                       out_mode = 'lin_layer')
    criticTarget2 = TransformerKnapsack(criticModelConfig, sacConfig.generat_link_number, 
                                       device=DEVICE, name = 'criticTarget2', 
                                       out_mode = 'lin_layer')
    env = KnapsackAssignmentEnv(actorModelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, MAIN_BATCH_SIZE, 
                                DEVICE)
    sacTrainer = FractionSACTrainer(INFOS, SAVE_PATH, MAIN_BATCH_SIZE, sacConfig, actorModel, 
                            criticLocal1, criticLocal2, criticTarget1, criticTarget2, 
                            DEVICE)
    return env, sacTrainer
    
def encoderMlpSACInitializer ():
    sacConfig = FractionSACConfig()
    modelConfig = TransformerKnapsackConfig(INSTANCE_OBS_SIZE, KNAPSACK_OBS_SIZE,
                                            opts.dim)
    actorModel = RNNMLPKnapsack(modelConfig, DEVICE)
    
    criticLocal1 = RNNMLPKnapsack(modelConfig, device=DEVICE, name = 'criticLocal1')
    criticLocal2 = RNNMLPKnapsack(modelConfig, device=DEVICE, name = 'criticLocal2')
    criticTarget1 = RNNMLPKnapsack(modelConfig, device=DEVICE, name = 'criticTarget1')
    criticTarget2 = RNNMLPKnapsack(modelConfig, device=DEVICE, name = 'criticTarget2')
    
    env = KnapsackAssignmentEnv(modelConfig.input_encode_dim, INFOS, NO_CHANGE_LONG, 
                                KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE, MAIN_BATCH_SIZE, DEVICE)
    
    sacTrainer = EncoderMlpSACTrainer(INFOS, SAVE_PATH, MAIN_BATCH_SIZE, sacConfig, actorModel, 
                                      criticLocal1, criticLocal2, criticTarget1, criticTarget2, 
                                      DEVICE)
    return env, sacTrainer 
                       
def plot_learning_curve(x, scores, figure_file, title, label):#TODO delete method
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg, 'C0', linewidth = 1, alpha = 0.5, label=label)
    plt.plot(np.convolve(running_avg, np.ones((3000,))/3000, mode='valid'), 'C0')
    plt.title(title)
    plt.savefig(figure_file)
    plt.show()


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
        transformer_trainer.train(opts.instances, 32)
    greedyScores = np.array(greedyScores)
    best_score = .8
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, MAIN_BATCH_SIZE)
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])

            externalObservation, _ = env.reset()
            done = False
            episodeInternalReward = torch.tensor (0.0)
            while not done:
                internalObservations, actions, accepted_actions, probs, values, \
                    internalRewards, steps = ppoTrainer.make_steps(
                        externalObservation, env.statePrepares)
                episodeInternalReward += internalRewards.sum().cpu()
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
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
                  'interanl_reward', float(episodeInternalReward))
    
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/fraction_ppo_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    label = 'scores'
    plot_learning_curve(x, score_history, figure_file, title, label)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/fraction_ppo_remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    label = 'remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title, label)
    
def ppo_train (env, ppoTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    '''if ppoTrainer.pretrain_need:
        transformer_trainer = TransformerTrainer(ppoTrainer.config.generat_link_number,
                                                 statePrepareList, ppoTrainer.actor_model, 
                                                 device=DEVICE)
        transformer_trainer.train(opts.instances, 32)'''
    greedyScores = np.array(greedyScores)
    best_score = .65
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, MAIN_BATCH_SIZE)
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])

            externalObservation, _ = env.reset()
            done = False
            while not done:
                internalObservatio, actions, accepted_acctions, sumProbs, sumVals, \
                    sumRewards, steps = ppoTrainer.make_steps(
                    externalObservation, env.statePrepares)
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
    figure_file = 'plots/ppo_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    label = 'scores'
    plot_learning_curve(x, score_history, figure_file, title, label)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/ppo_remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    label = 'remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title, label)
    
def fraction_sac_train (env, sacTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    greedyScores = np.array(greedyScores)
    best_score = .8
    score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        #batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, BATCH_SIZE)
        batchs = [np.array([0])]
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])
            
            externalObservation, _ = env.reset()
            done = False
            while not done: 
                internalObservations, actions, accepted_actions, rewards, steps = sacTrainer.steps(
                    externalObservation, env.statePrepares, done)
                externalObservation_, externalReward, done, info = env.step(accepted_actions)
                sacTrainer.save_step (externalObservation, internalObservations,
                                      actions, rewards, externalObservation_, steps, 
                                      done)
                sacTrainer.train()
                externalObservation = externalObservation_
                
            scores, remain_cap_ratios = env.final_score()
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])

            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score:
                best_score = avg_score
                sacTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios))
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/fraction_sac_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/fraction_sac_remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title)
    
def encoderMLP_sac_train (env, sacTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    greedyScores = np.array(greedyScores)
    best_reward = -1e4
    reward_history = []; score_history = []; remain_cap_history = []
    n_steps = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        #batchs = ppoTrainer.generate_batch(PROBLEMS_NUM, BATCH_SIZE)
        batchs = [np.array([0])]
        for batch in batchs:
            env.setStatePrepare(statePrepares[batch])
            
            externalObservation, _ = env.reset()
            done = False
            episodeInternalReward = torch.tensor (0.0)
            while not done:
                action, accepted_actions, internalReward = sacTrainer.make_steps(
                    externalObservation, env.statePrepares)
                episodeInternalReward += internalReward.sum().cpu()
                externalObservation_, externalReward, done, info = env.step(accepted_actions)
                sacTrainer.save_step (externalObservation, action, internalReward, 
                                      externalObservation_, done)
                n_steps += 1
                if n_steps % sacTrainer.config.internal_batch == 0:
                    sacTrainer.train()
                externalObservation = externalObservation_
                
            scores, remain_cap_ratios = env.final_score()
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])
            
            reward_history.append(float(episodeInternalReward))
            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_reward = np.mean(reward_history[-50:])
            avg_score = np.mean(score_history[-50:])

            if avg_reward > best_reward:
                best_reward  = avg_reward
                sacTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
                  'interanl_reward %.3f'%float(episodeInternalReward), 'avg reward %.3f' %avg_reward)
    
    x = [i+1 for i in range(len(reward_history))]
    figure_file = 'plots/encoderMLP_sac_reward.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, reward_history, figure_file, title)#TODO add visualization
            
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/encoderMLP_sac_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    plot_learning_curve(x, score_history, figure_file, title)#TODO add visualization
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/encoderMLP_sac_remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title)

def ppo_train_extra (env, ppoTrainer, statePrepareList, greedyScores):
    statePrepares = np.array(statePrepareList)
    
    greedyScores = np.array(greedyScores)
    best_reward = -1e4
    reward_history = []; score_history = []; remain_cap_history = []
    n_steps = 0
    n_steps1 = 0
    for i in tqdm(range(N_TRAIN_STEPS)):
        for batch in range(len(statePrepares)):
            env.setStatePrepare(statePrepares[0])

            externalObservation, _ = env.reset()
            done = False
            episodeNormalReward = torch.tensor (0.0)
            while not done:
                internalObservations, actions, accepted_action, probs, values, \
                    rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                           env.statePrepares, False, False)
                print(torch.cat(rewards,0).sum())
                episodeNormalReward += torch.cat(rewards,0).sum()
                externalObservation_, extraReward, done, info = env.step(accepted_action)
                
                if episodeNormalReward < -20: done = True
                ppoTrainer.memory.save_normal_step(externalObservation, actions, \
                                                   probs, values, rewards, done, \
                                                   steps, internalObservations)
                
                n_steps += 1
                if n_steps % ppoTrainer.config.normal_batch == 0:
                    ppoTrainer.train('normal')
                if ~(accepted_action == [-1,-1]).all():
                    #print(accepted_action)
                    n_steps1 +=1
                    ppoTrainer.memory.save_extra_step(externalObservation, actions, \
                                                      probs, values, extraReward, \
                                                      done, steps, internalObservations)   
                    if n_steps1 % ppoTrainer.config.extra_batch == 0:
                        ppoTrainer.train('extra')
                externalObservation = externalObservation_
                
            scores, remain_cap_ratios = env.final_score()
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])
            
            reward_history.append(float(episodeNormalReward))
            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_reward = np.mean(reward_history[-50:])
            avg_score = np.mean(score_history[-50:])
            
            if avg_reward > best_reward:
                best_reward  = avg_reward
                ppoTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
                  'interanl_reward %.3f'%float(episodeNormalReward), 'avg reward %.3f' %avg_reward)
    
    x = [i+1 for i in range(len(reward_history))]
    figure_file = 'plots/encoderMLP_ppo_reward.png'
    title = 'Running average of previous 50 scores'
    label = 'scores'
    plot_learning_curve(x, reward_history, figure_file, title, label)
    
    
    x = [i+1 for i in range(len(score_history))]
    figure_file = 'plots/encoderMLP_ppo_score_per_greedyScore.png'
    title = 'Running average of previous 50 scores'
    label = 'scores'
    plot_learning_curve(x, score_history, figure_file, title, label)
    
    x = [i+1 for i in range(len(remain_cap_history))]
    figure_file = 'plots/encoderMLP_ppo_remain_cap_ratio.png'
    title = 'Running average of previous 50 remain caps'
    label = 'remain caps'
    plot_learning_curve(x, remain_cap_history, figure_file, title, label)
                
if __name__ == '__main__':
    
    statePrepareList = dataInitializer()
    greedyScores = greedyAlgorithm(statePrepareList)
    
    #for algorithm_type in PPO_ALGORITHMS:
    #    for output_type in OUTPUT_TYPES:False, False
    env, ppoTrainer = ppoInitializer ('type3', 'EncoderPPOTrainer')
    ppo_train_extra(env, ppoTrainer, statePrepareList, greedyScores)
    