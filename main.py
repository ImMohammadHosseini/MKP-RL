
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
from data.dataProducer import multiObjectiveDimentionalKP
from configs.ppo_configs import PPOConfig
from configs.sac_configs import SACConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.models.EncoderMLP import EncoderMLPKnapsack, RNNMLPKnapsack
from src.models.critic_model import CriticNetwork1, CriticNetwork2
from src.data_structure.state_prepare import StatePrepare
#from src.transformer_trainer import TransformerTrainer
from solve_algorithms.RL.src.env import KnapsackAssignmentEnv
from solve_algorithms.RL.algorithm_initializer import algorithmInitializer
from solve_algorithms.random_select import RandomSelect
from solve_algorithms.greedy_select import GreedySelect



usage = "usage: python main.py -D <dim> -O <objective> -K <knapsaks> -M <mode> -N <instances> -T <trainMode>"

parser = optparse.OptionParser(usage=usage)
'''parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")'''
parser.add_option("-A", "--algorithm", action="store", dest="alg", 
                  default='EncoderSACTrainer')
parser.add_option("-P", "--output", action="store", dest="out", 
                  default='type3')

parser.add_option("-D", "--dim", action="store", dest="dim", default=2)
parser.add_option("-O", "--objective", action="store", dest="obj", default=1)
parser.add_option("-K", "--knapsaks", action="store", dest="kps", default=3)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=15)
parser.add_option("-M", "--mode", action="store", dest="mode", 
                  default='test')
parser.add_option("-T", "--trainMode", action="store", dest="trainMode", 
                  default='normal')

opts, args = parser.parse_args()

DEVICE = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
if opts.dim == 1: INFOS = {'CAP_LOW':120, 'CAP_HIGH':220, 'WEIGHT_LOW':10, 
                           'WEIGHT_HIGH':100, 'VALUE_LOW':3, 'VALUE_HIGH':200}
elif opts.dim > 1: INFOS = {'CAP_LOW':120+(10*(opts.dim-1)), 'CAP_HIGH':220+(10*(opts.dim-1)), 'WEIGHT_LOW':10, 
                           'WEIGHT_HIGH':100, 'VALUE_LOW':3, 'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = opts.instances#2 * KNAPSACK_OBS_SIZE
MAIN_BATCH_SIZE = 1
NO_CHANGE_LONG = 5#*BATCH_SIZE#int(1/8*(opts.instances // INSTANCE_OBS_SIZE))
PROBLEMS_NUM = 100000 if opts.mode == 'train' else 100 #1*MAIN_BATCH_SIZE
N_TRAIN_STEPS = 220002
#NEW_PROBLEM_PER_EPISODE = 10
N_TEST_STEPS = 10
SAVE_PATH = 'pretrained/save_models/'+opts.trainMode+'_dim_'+str(opts.dim)+'_obj_'+str(opts.obj)
RESULT_PATH = 'results/'+opts.mode+'/'+opts.trainMode+'_dim_'+str(opts.dim)+'_obj_'+str(opts.obj)
DATA_PATH = 'dataset/dim_'+str(opts.dim)+'_obj_'+str(opts.obj)+'/'+opts.mode+'/'



PPO_ALGORITHMS = ['EncoderPPOTrainer', 'FractionPPOTrainer', 'WholePPOTrainer']
SAC_ALGORITHMS = ['SACTrainer', 'FractionSACTrainer']
OTHER_ALGORITHMS = ['RandomSelect', 'GreedySelect']
INPUT_TYPES = ['type_1']
OUTPUT_TYPES = ['type_1', 'type_2', 'type_3']
ACTOR_MODEL_TYPES = ['Encoder_MLP', 'Transformer']
CRITIC_MODEL_TYPES = ['MLP']
TRAIN_PROCESS = ['normal_train', 'extra_train']

def dataInitializer ():
    if path.exists(DATA_PATH):
        instances_weights = [np.expand_dims(genfromtxt(DATA_PATH+'weights/'+str(i)+'.csv', delimiter=','),1) if opts.dim == 1 \
                             else genfromtxt(DATA_PATH+'weights/'+str(i)+'.csv', delimiter=',')\
                              for i in range(PROBLEMS_NUM)]
        instances_values = [np.expand_dims(genfromtxt(DATA_PATH+'values/'+str(i)+'.csv', delimiter=','),1) if opts.obj == 1
                            else genfromtxt(DATA_PATH+'values/'+str(i)+'.csv', delimiter=',')
                              for i in range(PROBLEMS_NUM)]
        
        ks_caps = [np.expand_dims(genfromtxt(DATA_PATH+'caps/'+str(i)+'.csv', delimiter=','),1) if opts.dim == 1
                   else genfromtxt(DATA_PATH+'caps/'+str(i)+'.csv', delimiter=',')
                              for i in range(PROBLEMS_NUM)]
        
    else:
        ks_caps, instances_weights, instances_values = multiObjectiveDimentionalKP(
            PROBLEMS_NUM, opts.dim, opts.obj, opts.instances, opts.kps, INFOS)
       
    if not path.exists(DATA_PATH):
        makedirs(DATA_PATH+'weights/')
        makedirs(DATA_PATH+'values/')
        makedirs(DATA_PATH+'caps/')
        for i,w in enumerate(instances_weights):
            np.savetxt(DATA_PATH+'weights/'+str(i)+'.csv', w, delimiter=",")
        for i,v in enumerate(instances_values):
            np.savetxt(DATA_PATH+'values/'+str(i)+'.csv', v, delimiter=",")
        for i,c in enumerate(ks_caps):
            np.savetxt(DATA_PATH+'caps/'+str(i)+'.csv', c, delimiter=",")
    
    
    return ks_caps, instances_weights, instances_values

def randomAlgorithm (statePrepare):
    #test_score_history = []
    #for i in tqdm(range(N_TEST_STEPS)):
    random_select = RandomSelect (statePrepare)
    random_select.test_step()#TODO
    
def greedyAlgorithm (c, w, v):
    for i in range (opts.obj): exec('score_history'+str(i)+'=[]')
    remain_cap_history = []; steps_history = []
    statePrepare = StatePrepare(INFOS)
    greedy_select = GreedySelect(opts.instances, statePrepare)

    for i in range(PROBLEMS_NUM):
        greedy_select.statePrepare.setProblem(c[i], w[i], v[i])
        greedy_select.reset()

        score, remain_cap_ratio, steps = greedy_select.test_step()
        for idx, val in enumerate(score):
            exec('score_history'+str(idx) +'.append(val)')        
        remain_cap_history.append(remain_cap_ratio)
        steps_history.append(steps)

        print('problem', i, 'scores', score, 'steps', steps, 
              'remain_cap_ratio %.3f'% np.mean(remain_cap_ratio))
    
        
    results_dict = {'remain_cap': remain_cap_history, 'steps': steps_history}
    for idx in range(opts.obj):results_dict['score'+str(idx)]=eval('score_history'+str(idx))
    if not path.exists(RESULT_PATH): makedirs(RESULT_PATH)
    with open(RESULT_PATH+'/greedy'+'_'+'.pickle', 'wb') as file:
        pickle.dump(results_dict, file)
        

 

def train_extra (env, trainer, flags, c, w, v):
    if 'SAC' in opts.alg:
        sac_train_extra(env, trainer, flags, c, w, v)
    elif 'PPO' in opts.alg:
        ppo_train_extra(env, trainer, flags, c, w, v)
        
def sac_train_extra (env, sacTrainer, flags, c, w, v):
    best_reward = 35
    
    #reward_history = []; score_history = [[] for i in range(opts.obj)]; 
    #remain_cap_history = []; steps_history = []
    result_path=RESULT_PATH+'/'+opts.alg+'_'+opts.out+'.pickle'
    print(result_path)
    with open(result_path, 'rb') as handle:
        results = pickle.load(handle)
    reward_history = results['reward']
    score_history = results['score'] 
                     
    remain_cap_history = results['remain_cap']
    steps_history = results['steps'] 
    trainer.memory.load_buffer()
    #TODO delete 
    #n_steps = 0
    #n_steps1 = 0
    repeated = 0
    #addition_steps = sacTrainer.config.generat_link_number if flags[0] else 1
    change_problem = True
    for i in tqdm(range(200000, N_TRAIN_STEPS)):
        #for batch in range(len(statePrepares)):
        if change_problem:
            pn = np.random.randint(PROBLEMS_NUM)
            env.statePrepare.setProblem(c[pn], w[pn], v[pn])
        
        externalObservation, _ = env.reset(change_problem)
        done = False
        change_problem = True
        episodeNormalReward = torch.tensor (0.0)
        step_num = 0
        while not done:
            #print(externalObservation)
            internalObservations, actions, accepted_action, rewards, steps\
                = sacTrainer.make_steps(externalObservation, env.statePrepare, 
                                        flags[0], flags[1])
            #print(rewards)
            #print(torch.cat(rewards,0).sum())
            episodeNormalReward += torch.cat(rewards,0).sum()
            externalObservation_, extraReward, done, info = env.step(accepted_action)
            sacTrainer.memory.save_normal_step(externalObservation, externalObservation_, \
                                               actions, rewards, done, steps, \
                                               internalObservations)
            step_num += 1

            if episodeNormalReward < -10: 
                done = True
                if episodeNormalReward < 0 and repeated < 50:
                    change_problem = False
                    repeated += 1
                else:repeated = 0
            
            sacTrainer.train('normal')
            if opts.trainMode == 'extra' and ~(accepted_action == [-1,-1]).all():
                extraReward = rewards
                sacTrainer.memory.save_extra_step(externalObservation, actions, \
                                                  extraReward, \
                                                  done, steps, internalObservations)   
                sacTrainer.train('extra')
            externalObservation = externalObservation_
            
        scores, remain_cap_ratios = env.final_score()
        #batch_score_per_grredy = scores/greedyScores[batch]
        
        reward_history.append(float(episodeNormalReward))
        [score_history[_].append(scores[_]) for _ in range(opts.obj)]
        steps_history.append(step_num)
        
        remain_cap_history.append(remain_cap_ratios)
        avg_reward = np.mean(reward_history[-100:])
        #print([score_history[_][-2:] for _ in range(opts.obj)])
        avg_score = np.mean([score_history[_][-100:] for _ in range(opts.obj)], 1)
        #print(avg_score)
            
        if avg_reward > best_reward:
            best_reward  = avg_reward
            sacTrainer.save_models()
            #TODO ooo check the critic net
        print('episode', i, 'scores', scores, 'avg scores', avg_score,
              'steps', step_num, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
              'interanl_reward %.3f'%float(episodeNormalReward), 'avg reward %.3f' %avg_reward)
        #print(sacTrainer.dd)
        if i % 500 == 0:
            results_dict = {'reward': reward_history, 'score': score_history, 
                            'remain_cap': remain_cap_history, 'steps': steps_history}
            if not path.exists(RESULT_PATH): makedirs(RESULT_PATH)
            with open(RESULT_PATH+'/'+opts.alg+'_'+opts.out+'.pickle', 'wb') as file:
                pickle.dump(results_dict, file)
def ppo_train_extra (env, ppoTrainer, flags, c, w, v):
    #statePrepares = np.array(statePrepareList)
    
    #greedyScores = np.array(greedyScores)
    best_reward = -1e4
    
    reward_history = []; score_history = []; remain_cap_history = []; steps_history = []
    '''result_path='results/train/normal_dim_2_obj_1/EncoderPPOTrainer_type3.pickle'
    with open(result_path, 'rb') as handle:
        results = pickle.load(handle)
    reward_history = results['reward']
    score_history = results['score'] 
    remain_cap_history = results['remain_cap']
    steps_history = results['steps'] '''
    #TODO delete 
    n_steps = 0
    n_steps1 = 0
    repeated = 0
    addition_steps = ppoTrainer.config.generat_link_number if flags[0] else 1
    change_problem = True
    for i in tqdm(range(N_TRAIN_STEPS)):
        #for batch in range(len(statePrepares)):
        if change_problem:
            pn = np.random.randint(PROBLEMS_NUM)
            env.statePrepare.setProblem(c[1], w[1], v[1])
        
        externalObservation, _ = env.reset(change_problem)
        done = False
        change_problem = True
        episodeNormalReward = torch.tensor (0.0)
        step_num = 0
        while not done:
            internalObservations, actions, accepted_action, probs, values, \
                rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                       env.statePrepare, flags[0], flags[1])
            #print(torch.cat(rewards,0).sum())
            episodeNormalReward += torch.cat(rewards,0).sum()
            externalObservation_, extraReward, done, info = env.step(accepted_action)
            
            if episodeNormalReward < -50: 
                done = True
                if (episodeNormalReward < 0 or step_num > np.mean(score_history[-100:])) and repeated < 50:
                    change_problem = False
                    repeated += 1
                else:repeated = 0
            
            ppoTrainer.memory.save_normal_step(externalObservation, actions, \
                                               probs, values, rewards, done, \
                                               steps, internalObservations)
            
            n_steps += addition_steps
            step_num += addition_steps
            if n_steps % ppoTrainer.config.normal_batch == 0:
                ppoTrainer.train('normal')
            if opts.trainMode == 'extra' and ~(accepted_action == [-1,-1]).all():
                #print(accepted_action)
                n_steps1 += addition_steps
                extraReward = rewards
                ppoTrainer.memory.save_extra_step(externalObservation, actions, \
                                                  probs, values, extraReward, \
                                                  done, steps, internalObservations)   
                if n_steps1 % ppoTrainer.config.extra_batch == 0:
                    ppoTrainer.train('extra')
            externalObservation = externalObservation_
            
        scores, remain_cap_ratios = env.final_score()
        
        #batch_score_per_grredy = scores/greedyScores[batch]
        
        reward_history.append(float(episodeNormalReward))
        score_history.append(scores[0])#TODO change score for more obj#batch_score_per_grredy)
        steps_history.append(step_num)
        
        #print(score_history)
        remain_cap_history.append(remain_cap_ratios)
        avg_reward = np.mean(reward_history[-100:])
        avg_score = np.mean(score_history[-100:], 0)
        #print(avg_score)
            
        if avg_reward > best_reward:
            best_reward  = avg_reward
            ppoTrainer.save_models()
            #TODO ooo check the critic net
        print('episode', i, 'scores', scores, 'avg scores', avg_score,
              'steps', step_num, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
              'interanl_reward %.3f'%float(episodeNormalReward), 'avg reward %.3f' %avg_reward)
    
        if i % 1000 == 0:
            results_dict = {'reward': reward_history, 'score': score_history, 
                            'remain_cap': remain_cap_history, 'steps': steps_history}
            if not path.exists(RESULT_PATH): makedirs(RESULT_PATH)
            with open(RESULT_PATH+'/'+opts.alg+'_'+opts.out+'.pickle', 'wb') as file:
                pickle.dump(results_dict, file)
            

    
def ppo_test(env, ppoTrainer, flags, c, w, v):
    
    score_history = []; remain_cap_history = []; steps_history = []
    
    for i in range (opts.obj): exec('score_history'+str(i)+'=[]')
    #print(eval('score_history0'))
    for i in range(PROBLEMS_NUM):
        env.statePrepare.setProblem(c[i], w[i], v[i])
        
        
        externalObservation, _ = env.reset()
        done = False
        step_num = 0
        while not done:
            internalObservations, actions, accepted_action, probs, values, \
                rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                       env.statePrepare, flags[0], flags[1])
            externalObservation_, extraReward, done, info = env.step(accepted_action)
            step_num += 1
            externalObservation = externalObservation_

            #print(actions)
            #print(rewards)

        scores, remain_cap_ratios = env.final_score()
        
        #score_history.append(scores)
        for idx, val in enumerate(scores):
            exec('score_history'+str(idx)+'.append(val)')
        steps_history.append(step_num)
        remain_cap_history.append(remain_cap_ratios)
        
        print('problem', i, 'scores', scores, 'steps', step_num, 
              'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),)
    for idx in range(opts.obj):
        print('hh')
        print(eval('score_history'+str(idx)))
    
    results_dict = {'remain_cap': remain_cap_history, 'steps': steps_history}
    for idx in range(opts.obj):results_dict['score'+str(idx)]=eval('score_history'+str(idx))
    '''results_dict = {'score'+str(idx):eval('score_history'+str(idx)) for idx in range(opts.obj)}    
    results_dict['remain_cap']= remain_cap_history 
    results_dict['steps']= steps_history'''
    if not path.exists(RESULT_PATH): makedirs(RESULT_PATH)
    with open(RESULT_PATH+'/'+opts.alg+'_'+opts.out+'.pickle', 'wb') as file:
        pickle.dump(results_dict, file)

if __name__ == '__main__':
    
    c, w, v = dataInitializer()
    #greedyScores = greedyAlgorithm(statePrepareList)
    
    #for algorithm_type in PPO_ALGORITHMS:
    #    for output_type in OUTPUT_TYPES:False, False
    env, trainer, flags = algorithmInitializer (opts.alg)
    if opts.mode == 'train':
        train_extra(env, trainer, flags, c, w, v)
    elif opts.mode == 'test':
        greedyAlgorithm(c, w, v)
        ppo_test(env, trainer, flags, c, w, v)
    