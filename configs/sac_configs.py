
"""

"""


from dataclasses import dataclass

@dataclass
class SACConfig(object):
    """
    Configuration class for PPOTrainer
    """
    def __init__ (
        self,
        generat_link_number: int = 10,
        sac_batch_size: int = 256,
        buffer_size=int(1e4),

        
        
        
        gradient_steps_per_itr,
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
        spatial_regularization_eps=1.
    ):
        self.generat_link_number = generat_link_number
        self.sac_batch_size = sac_batch_size
        self.buffer_size = buffer_size


        