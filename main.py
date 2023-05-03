
"""

"""
import optparse
import torch
from data.dataProducer import multipleKnapSackData
from data.prepare import getSituations, getPrompt
from configs.ppo_configs import PPOConfig
from src.models.transformer import TransformerKnapsack
from RL.src.env import KnapsackAssignmentEnv
from RL.ppo_trainer import PPOTrainer

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multiObjectiveDimentional',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-D", "--dim", action="store", dest="dimention", default=5)
parser.add_option("-M", "--knapsaks", action="store", dest="kps", default=600)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=10000)
opts, args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFOS = {'CAP_LOW':5, 'CAP_HIGH':150, 'WEIGHT_LOW':2, 'WEIGHT_HIGH':150,
         'VALUE_LOW':1, 'VALUE_HIGH':200}
KNAPSACK_BATCH_SIZE = 60
INSTANCE_BATCH_SIZE = 3 * KNAPSACK_BATCH_SIZE

def datainitializer ():
    if opts.var == 'multipleKnapsack':
       c, w, v = multipleKnapSackData(opts.kps, opts.instances, INFOS)
    elif opts.var == 'multipleKnapsack':
        pass
    elif opts.var == 'multiObjectiveDimentional':
        c, w, v = multipleKnapSackData(opts.dim, opts.kps, opts.instances, INFOS)
    
    situations = getSituations(c, w, opts.dim)
    prompt = getPrompt(situations)
    return situations, prompt, (c, w, v)
    
def rlinit ():
    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
    )   
    model = TransformerKnapsack
    env = KnapsackAssignmentEnv
    ppo_trainer = PPOTrainer()
    
    return env, ppo_trainer

def train ():
    pass

if __name__ == '__main__':
    situations, prompt, datas = datainitializer ()
    
    