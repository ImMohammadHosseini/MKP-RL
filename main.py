
"""

"""
import optparse

from utils.dataProducer import multipleKnapSackData

usage = "usage: python main.py -V <variation> -M <knapsaks> -N <instances>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-V", "--variation", action="store", dest="var", 
                  default='multipleKnapsack',
                  help="variation is multipleKnapsack, multi_dimentional, \
                      multiObjectiveDimentional")
parser.add_option("-M", "--knapsaks", action="store", dest="kps", default=60)
parser.add_option("-N", "--instances", action="store", dest="instances", 
                  default=3000)
opts, args = parser.parse_args()

def datainitializer ():
    if opts.var == 'multipleKnapsack':
       c, w, v = multipleKnapSackData(opts.kps, opts.instances)
    elif opts.var == 'multipleKnapsack':
        pass
    elif opts.var == 'multiObjectiveDimentional':
        pass

if __name__ == '__main__':
    
