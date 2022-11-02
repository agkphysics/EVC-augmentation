from dataclasses import dataclass, field
import csv
from typing import List

from omegaconf import OmegaConf


@dataclass
class HParams:
    ################################
    # Experiment Parameters        #
    ################################
    epochs: int = 70
    iters_per_checkpoint: int = 2000
    iters_per_validate: int = 100
    seed: int = 1234
    dynamic_loss_scaling: bool = True
    fp16_run: bool = False
    distributed_run: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = False

    ################################
    # Data Parameters              #
    ################################
    training_list: str = ""
    validation_list: str = ""
    mel_mean_std: str = ""

    phones_csv: str = ""
    emo_csv: str = ""

    # emotion_A: str = "Angry"
    # emotion_B: str = "Happy"
    # emotion_C: str = "Sad"
    # emotion_D: str = "Neutral"
    # emotion_E: str = "Surprise"

    # a_embedding_path: str = 'zero_embeddings.npy'
    # b_embedding_path: str = 'zero_embeddings.npy'
    # c_embedding_path: str = 'zero_embeddings.npy'
    # d_embedding_path: str = 'zero_embeddings.npy'
    # e_embedding_path: str = 'zero_embeddings.npy'

    # a_embedding_path: str = "/home/akee511/src/seq2seq-EVC/conversion/outdir/embeddings/anger.npy"
    # b_embedding_path: str = "/home/akee511/src/seq2seq-EVC/conversion/outdir/embeddings/happiness.npy"
    # c_embedding_path: str = "/home/akee511/src/seq2seq-EVC/conversion/outdir/embeddings/sadness.npy"
    # d_embedding_path: str = "/home/akee511/src/seq2seq-EVC/conversion/outdir/embeddings/neutral.npy"
    # e_embedding_path: str = "/home/akee511/src/seq2seq-EVC/conversion/outdir/embeddings/surprise.npy"

    emo_embedding_dir: str = ""
    emo_list: List[str] = field(default_factory=list)

    emo_inference_weight: float = 1.0

    spk_embedding_dir: str = ""
    spk_list: List[str] = field(default_factory=list)

    ################################
    # Data Parameters              #
    ################################
    n_mel_channels: int = 80
    n_spc_channels: int = 1025
    n_symbols: int = 41  #
    pretrain_n_speakers: int = 99  #
    n_emotions: int = 5
    predict_spectrogram: bool = False

    ################################
    # Model Parameters             #
    ################################

    symbols_embedding_dim: int = 512

    # Text Encoder parameters
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512
    text_encoder_dropout: float = 0.5

    # Audio Encoder parameters
    spemb_input: bool = False
    n_frames_per_step_encoder: int = 2
    audio_encoder_hidden_dim: int = 512
    AE_attention_dim: int = 128
    AE_attention_location_n_filters: int = 32
    AE_attention_location_kernel_size: int = 51
    beam_width: int = 10

    # hidden activation
    # relu linear tanh
    hidden_activation: str = "tanh"

    # Speaker Encoder parameters
    emotion_encoder_hidden_dim: int = 256
    emotion_encoder_dropout: float = 0.2
    emotion_embedding_dim: int = 128

    # Text Classifier parameters
    # text_classifier_hidden_dim=256

    # Speaker Classifier parameters
    SC_hidden_dim: int = 512
    SC_n_convolutions: int = 3
    SC_kernel_size: int = 1

    # Decoder parameters
    feed_back_last: bool = True
    n_frames_per_step_decoder: int = 2
    decoder_rnn_dim: int = 512
    prenet_dim: List[int] = field(default_factory=lambda: [256, 256])
    max_decoder_steps: int = 1000
    stop_threshold: float = 0.5

    # Attention parameters
    attention_rnn_dim: int = 512
    attention_dim: int = 128

    # Location Layer parameters
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 17

    # PostNet parameters
    postnet_n_convolutions: int = 5
    postnet_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_dropout: float = 0.5

    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 5.0
    batch_size: int = 16
    warmup: int = 7
    decay_rate: float = 0.5
    decay_every: int = 7

    contrastive_loss_w: float = 30.0
    emotion_encoder_loss_w: float = 0.0
    text_classifier_loss_w: float = 1.0
    emotion_adversial_loss_w: float = 0.2
    emotion_classifier_loss_w: float = 1.0
    ser_loss_w: float = 0.1
    # ce_loss: bool = False


def create_hparams(override_str: str = None) -> HParams:
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = OmegaConf.structured(HParams)
    if override_str:
        overrides = next(csv.reader([override_str]))
        overrides = OmegaConf.from_dotlist(overrides)
        hparams = OmegaConf.merge(hparams, overrides)
    return hparams
