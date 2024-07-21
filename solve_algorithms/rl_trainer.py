
import numpy as np
from os import path, makedirs
import pickle
import torch
from tqdm import tqdm




def train_extra (env, trainer, train_steps, problem_num, flags, c, w, v, r_path, opts):
    if 'SAC' in opts.alg:
        sac_train_extra(env, trainer, train_steps, problem_num, flags, c, w, v, r_path, opts)
    elif 'PPO' in opts.alg:
        ppo_train_extra(env, trainer, train_steps, problem_num, flags, c, w, v, r_path, opts)
        
def sac_train_extra (env, sacTrainer, train_steps, problem_num, flags, c, w, v, r_path, opts):
    best_reward = 35
    
    #reward_history = []; score_history = [[] for i in range(opts.obj)]; 
    #remain_cap_history = []; steps_history = []
    result_path=r_path+'/'+opts.alg+'_'+opts.out+'.pickle'
    print(result_path)
    with open(result_path, 'rb') as handle:
        results = pickle.load(handle)
    reward_history = results['reward']
    score_history = results['score'] 
                     
    remain_cap_history = results['remain_cap']
    steps_history = results['steps'] 
    sacTrainer.memory.load_buffer()
    #TODO delete 
    #n_steps = 0
    #n_steps1 = 0
    repeated = 0
    #addition_steps = sacTrainer.config.generat_link_number if flags[0] else 1
    change_problem = True
    for i in tqdm(range(200000, train_steps)):
        #for batch in range(len(statePrepares)):
        if change_problem:
            pn = np.random.randint(problem_num)
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
            if not path.exists(r_path): makedirs(r_path)
            with open(r_path+'/'+opts.alg+'_'+opts.out+'.pickle', 'wb') as file:
                pickle.dump(results_dict, file)
                
def ppo_train_extra (env, ppoTrainer, train_steps, problem_num, flags, c, w, v, r_path, opts):
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
    for i in tqdm(range(train_steps)):
        #for batch in range(len(statePrepares)):
        if change_problem:
            pn = np.random.randint(problem_num)
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
            if not path.exists(r_path): makedirs(r_path)
            with open(r_path+'/'+opts.alg+'_'+opts.out+'.pickle', 'wb') as file:
                pickle.dump(results_dict, file)