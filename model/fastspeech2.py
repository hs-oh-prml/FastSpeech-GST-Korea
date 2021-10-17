import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from GST import GST

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(model_config["transformer"]["decoder_hidden"], preprocess_config["preprocessing"]["mel"]["n_mel_channels"],)
        self.postnet = PostNet()

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        ref_mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)
        e_output = self.encoder(texts, src_masks)

        e_output = e_output

        (
            v_output,
            p_predictions, e_predictions, log_d_predictions,
            d_rounded, mel_lens, mel_masks,
        ) = self.variance_adaptor(
            e_output,
            src_masks, mel_masks, max_mel_len,
            p_targets, e_targets, d_targets,
            p_control, e_control, d_control,
        )
        output, mel_masks = self.decoder(v_output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens
        )
