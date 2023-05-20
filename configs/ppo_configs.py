
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
        generat_link_number: int = 100,
        internal_batch: int = 20,
        ppo_batch_size: int = 5,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        cliprange: float = 0.2,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0001,
        seed: float = 0,
        
    ):
        assert generat_link_number >= internal_batch
        self.generat_link_number = generat_link_number #number of links generated in on external observation
        self.internal_batch = internal_batch #number of links generated to optimize model
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cliprange = cliprange
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed
        
    