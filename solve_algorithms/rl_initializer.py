

from .RL.fraction_sac_trainer import FractionSACTrainer
from .RL.encoder_ppo_trainer import (
    EncoderPPOTrainer_t2, 
    EncoderPPOTrainer_t3
    )
from .RL.encoder_sac_trainer import EncoderSACTrainer_t3
from .RL.whole_ppo_trainer import (
    WholePPOTrainer_t1,
    WholePPOTrainer_t2,
    WholePPOTrainer_t3
    )
from .RL.fraction_ppo_trainer import (
    FractionPPOTrainer_t1,
    FractionPPOTrainer_t2,
    FractionPPOTrainer_t3
    )

from configs.ppo_configs import PPOConfig
from configs.sac_configs import SACConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.data_structure.state_prepare import StatePrepare

from src.models.transformer import TransformerKnapsack
from src.models.EncoderMLP import EncoderMLPKnapsack, RNNMLPKnapsack
from src.models.critic_model import CriticNetwork1, CriticNetwork2
from .RL.src.env import KnapsackAssignmentEnv


def algorithmInitializer (instance_obs_size, kp_obs_size, info, opts, no_change_long, 
                          save_path, DEVICE):
    if 'SAC' in opts.alg:
        return sacInitializer(instance_obs_size, kp_obs_size, info, opts, no_change_long, 
                              save_path, DEVICE)
    elif 'PPO' in opts.alg:
        return ppoInitializer(instance_obs_size, kp_obs_size, info, opts, no_change_long, 
                              save_path, DEVICE)
    
    
    
   
def sacInitializer (instance_obs_size, kp_obs_size, info, opts, no_change_long, 
                    save_path, DEVICE):
    sacConfig = SACConfig(generat_link_number=1) if opts.alg == 'EncoderSACTrainer' else SACConfig()
    modelConfig = TransformerKnapsackConfig(instance_obs_size, kp_obs_size,
                                            opts.dim+opts.obj, DEVICE, sacConfig.generat_link_number,opts.dim+1)
    env = KnapsackAssignmentEnv(opts.dim+opts.obj, info, no_change_long, 
                                kp_obs_size, instance_obs_size,device=DEVICE)
    statePrepare = StatePrepare(info)
    env.setStatePrepare(statePrepare)
    
    if opts.alg == 'EncoderSACTrainer':
        actorModel = EncoderMLPKnapsack(modelConfig, opts.out, device=DEVICE)
        critic_local1 = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                       device=DEVICE, out_put_dim=kp_obs_size*instance_obs_size,
                                       name='critic_local1')
        critic_local2 = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                       device=DEVICE, out_put_dim=kp_obs_size*instance_obs_size,
                                       name='critic_local2')
        critic_target1 = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                        device=DEVICE, out_put_dim=kp_obs_size*instance_obs_size,
                                        name='critic_target1')
        critic_target2 = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                        device=DEVICE, out_put_dim=kp_obs_size*instance_obs_size,
                                        name='critic_target2')
        if opts.out == 'type2':
            pass

        elif opts.out == 'type3': 
            sacTrainer = EncoderSACTrainer_t3(info, save_path, sacConfig, opts.dim, opts.obj,
                                              actorModel, critic_local1, critic_local2,
                                              critic_target1, critic_target2)
            flags = [True, False]
            
    elif opts.alg == 'FractionSACTrainer':
        pass
    elif opts.alg == 'WholeSACTrainer':
        pass
    
    return env, sacTrainer, flags

        
def ppoInitializer (instance_obs_size, kp_obs_size, info, opts, no_change_long, 
                    save_path, DEVICE):
    ppoConfig = PPOConfig(generat_link_number=1) if opts.alg == 'EncoderPPOTrainer' else PPOConfig()
    modelConfig = TransformerKnapsackConfig(instance_obs_size, kp_obs_size,
                                            opts.dim+opts.obj, DEVICE, ppoConfig.generat_link_number,opts.dim+1)
    
    env = KnapsackAssignmentEnv(opts.dim+opts.obj, info, no_change_long, 
                                kp_obs_size, instance_obs_size,device=DEVICE)
    statePrepare = StatePrepare(info)
    env.setStatePrepare(statePrepare)
    
    if opts.alg == 'EncoderPPOTrainer':
        actorModel = EncoderMLPKnapsack(modelConfig, opts.out, device=DEVICE)
        normalCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork2(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          device=DEVICE, name='extraCriticModel')
        
        if opts.out == 'type2':
            ppoTrainer = EncoderPPOTrainer_t2(info, save_path, ppoConfig, opts.dim, 
                                              actorModel, normalCriticModel, extraCriticModel)
            flags = [True, False]

        elif opts.out == 'type3': 
            ppoTrainer = EncoderPPOTrainer_t3(info, save_path, ppoConfig, opts.dim, 
                                              actorModel, normalCriticModel, extraCriticModel)
            flags = [True, False]
        
    elif opts.alg == 'FractionPPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, opts.out, device=DEVICE)
        normalCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                          device=DEVICE, name='extraCriticModel')
        
        if opts.out == 'type1': 
            ppoTrainer = FractionPPOTrainer_t1(info, save_path, ppoConfig, opts.dim, 
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
            
        elif opts.out == 'type2':
            ppoTrainer = FractionPPOTrainer_t2(info, save_path, ppoConfig, opts.dim,
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
            
        elif opts.out == 'type3':
            ppoTrainer = FractionPPOTrainer_t3(info, save_path, ppoConfig, opts.dim, 
                                               actorModel, normalCriticModel, extraCriticModel)
            flags = [True, True]
        
    elif opts.alg == 'WholePPOTrainer':
        actorModel = TransformerKnapsack(modelConfig, opts.out, device=DEVICE)
        normalCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                           (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                           device=DEVICE, name='normalCriticModel')
        extraCriticModel = CriticNetwork1(modelConfig.max_length, modelConfig.input_encode_dim, 
                                          (ppoConfig.generat_link_number+1), modelConfig.input_decode_dim,
                                          device=DEVICE, name='extraCriticModel')  
        if opts.out == 'type1': 
            ppoTrainer = WholePPOTrainer_t1(info, save_path, ppoConfig, opts.dim,
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]
            
        elif opts.out == 'type2': 
            ppoTrainer = WholePPOTrainer_t2(info, save_path, ppoConfig, opts.dim, 
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]
            
        elif opts.out == 'type3':
            ppoTrainer = WholePPOTrainer_t3(info, save_path, ppoConfig, opts.dim, 
                                            actorModel, normalCriticModel, extraCriticModel)
            flags = [False, True]

    return env, ppoTrainer, flags