
"""

"""
import optparse
import torch
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from configs.ppo_configs import PPOConfig
from configs.transformers_model_configs import TransformerKnapsackConfig
from src.models.transformer import TransformerKnapsack
from src.data_structure.state_prepare import StatePrepare
from RL.src.env import KnapsackAssignmentEnv
from RL.ppo_trainer import PPOTrainer

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dimention", default=5)
parser.add_option("-M", "--knapsaks", action="store", dest="kps", default=60)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=3000)
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':150, 'CAP_HIGH':400, 'WEIGHT_LOW':10, 'WEIGHT_HIGH':100,
         'VALUE_LOW':1, 'VALUE_HIGH':200}

KNAPSACK_OBS_SIZE = opts.kps
INSTANCE_OBS_SIZE = 2 * KNAPSACK_OBS_SIZE

def dataInitializer ():
    if opts.var == 'multipleKnapsack':
       c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
    elif opts.var == 'multipleKnapsack':
        pass
    elif opts.var == 'multiObjectiveDimentional':
        c, w, v = multiObjectiveDimentional(opts.dim, opts.kps, opts.instances, INFOS)
    
    statePrepare = StatePrepare(c, w, v, KNAPSACK_OBS_SIZE, INSTANCE_OBS_SIZE)
    
    return statePrepare
    
def rlInitializer (statePrepare):
    ppoConfig = PPOConfig()
    modelConfig = TransformerKnapsackConfig()#TODO check config file
    model = TransformerKnapsack(modelConfig)
    env = KnapsackAssignmentEnv(ppoConfig, INFOS, statePrepare, DEVICE)
    ppoTrainer = PPOTrainer(ppoConfig, model,)
    
    return env, ppoTrainer

def train (env, ppoTrainer):
    pass

if __name__ == '__main__':
    statePrepare = dataInitializer()
    env, ppoTrainer = rlInitializer(statePrepare)
    
    