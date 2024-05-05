
"""

"""


from dataclasses import dataclass

@dataclass
class SACConfig(object):
    """
    Configuration class for SACTrainer
    """
    def __init__ (
        self,
        generat_link_number: int = 5,
        normal_batch_size: int = 32,
        extra_batch_size: int = 32,
        buffer_size: int = int(1e6),
        alpha_initial: float = 1.,
        discount_rate: float = 0.99,
        actor_lr: float = 1e-5,
        critic_lr: float = 1e-5,
        alpha_lr: float = 1e-5,
        tau: float = 0.01,
    ):
        '''gradient_steps_per_itr,
        max_episode_length_eval=None,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        steps_per_epoch=1,
        num_evaluation_episodes=10,
        eval_env=None,
        use_deterministic_evaluation=True,
        temporal_regularization_factor=0.,
        spatial_regularization_factor=0.,
        spatial_regularization_eps=1.'''
    
        self.generat_link_number = generat_link_number
        self.normal_batch_size = normal_batch_size
        self.extra_batch_size = extra_batch_size
        self.buffer_size = buffer_size
        self.alpha_initial = alpha_initial
        self.discount_rate = discount_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.tau = tau
        