
"""

"""

from dataclasses import dataclass

@dataclass
class PPOConfig(object):
    """
    Configuration class for PPOTrainer
    """
    def __init__ (
        self,
        generat_link_number: int = 5,
        normal_batch: int = 32,
        ppo_normal_batch_size: int = 8,
        ppo_epochs: int = 10,
        extra_batch: int = 32,
        ppo_extra_batch_size: int = 8,
        ppo_extra_epochs: int = 5,
        gamma: float = 0.9,
        gae_lambda: float = 0.97,
        cliprange: float = 0.2,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        seed: int = 0,
        
    ):
        #assert generat_link_number >= internal_batch
        self.generat_link_number = generat_link_number #number of links generated in on external observation
        self.normal_batch = normal_batch #number of links generated to optimize model
        self.ppo_normal_batch_size = ppo_normal_batch_size
        self.ppo_epochs = ppo_epochs
        self.extra_batch = extra_batch
        self.ppo_extra_batch_size = ppo_extra_batch_size
        self.ppo_extra_epochs = ppo_extra_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        
    