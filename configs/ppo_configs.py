
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
        generat_link_number: int = 10,
        internal_batch: int = 80,
        ppo_batch_size: int = 20,
        ppo_epochs: int = 5,
        ppo_external_batch_size: int = 10,
        ppo_external_epochs: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.97,
        cliprange: float = 0.2,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        seed: int = 0,
        
    ):
        #assert generat_link_number >= internal_batch
        self.generat_link_number = generat_link_number #number of links generated in on external observation
        self.internal_batch = internal_batch #number of links generated to optimize model
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epochs = ppo_epochs
        self.ppo_external_batch_size = ppo_external_batch_size
        self.ppo_external_epochs = ppo_external_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        
    