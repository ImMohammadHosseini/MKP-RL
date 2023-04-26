
"""

"""

from transformers import PretrainedConfig


class TransformerKnapsackConfig (PretrainedConfig):
    model_type = "TransformerKnapsack"

    def __init__(
        self,
        num_instances: int = 85,#TODO
        d_model: int = 128,
        nhead: int = 2,
        d_hid: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.0,
        batch_first: bool = True,
        init_pick_predictor: bool = False,
        init_place_predictor: bool = False,
        n_input_dim: int = 3,
        place_dim: int = 7,
        category_embed_size: int = 32,
        pose_embed_size: int = 128,
        temporal_embed_size: int = 64,
        marker_embed_size: int = 32,
        **kwargs,
    ) -> None:
        self.num_instances = num_instances
        self.init_pick_predictor = init_pick_predictor
        self.init_place_predictor = init_place_predictor
        self.d_model = d_model
        self.batch_first = batch_first
        self.n_input_dim = n_input_dim
        self.nhead = nhead
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.place_dim = place_dim
        self.category_embed_size = category_embed_size
        self.pose_embed_size = pose_embed_size
        self.temporal_embed_size = temporal_embed_size
        self.marker_embed_size = marker_embed_size
        assert (
            category_embed_size
            + pose_embed_size
            + temporal_embed_size
            + marker_embed_size
            == d_model
        )
        super(TransformerKnapsackConfig, self).__init__(**kwargs)