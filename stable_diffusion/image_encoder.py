# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import tensorflow as tf
from keras import layers, Sequential,utils
from .layers import GroupNormalization, AttentionBlock, PaddedConv2D, ResnetBlock
from .ckpt_loader import load_weights_from_file, CKPT_MAPPING


class ImageEncoder(Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""

    def __init__(self, ckpt_path=None):
        super().__init__(
            [
                layers.Input((None, None, 3)),
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding=((0, 1), (0, 1)), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                GroupNormalization(epsilon=1e-5),
                layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),
                PaddedConv2D(8, 1),
                layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0] * 0.18215),
            ])
        origin = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors"
        ckpt_mapping = CKPT_MAPPING["encoder"]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = utils.get_file(origin=origin, fname="image_encoder_sd15.safetensors")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping)
