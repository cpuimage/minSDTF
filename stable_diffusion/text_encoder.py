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

import numpy as np
import tensorflow as tf

from .ckpt_loader import load_weights_from_file


class CLIPEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim=49408, output_dim=768, max_length=77, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = tf.keras.layers.Embedding(input_dim, output_dim, name="token_embedding")
        self.position_embedding = tf.keras.layers.Embedding(max_length, output_dim, name="position_embedding")

    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        return tokens + positions


class CLIPEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm1")
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True, name="self_attn")
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm2")
        self.fc1 = tf.keras.layers.Dense(embed_dim * 4, name="mlp.fc1")
        self.fc2 = tf.keras.layers.Dense(embed_dim, name="mlp.fc2")
        self.activation = activation

    def call(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class CLIPAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = tf.keras.layers.Dense(self.embed_dim, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.embed_dim, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dim, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dim, name="out_proj")

    def reshape_states(self, x, sequence_length, batch_size):
        x = tf.reshape(
            x, (batch_size, sequence_length, self.num_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))  # bs, heads, sequence_length, head_dim

    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = inputs.get_shape().as_list()[1]
            attention_mask = tf.cast(np.triu(np.ones((1, 1, length, length), dtype="float32") * -np.inf, k=1),
                                     dtype=self.compute_dtype)
        _, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)
        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)
        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ tf.transpose(key_states, (0, 2, 1))
        attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))
        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states
        attn_output = tf.reshape(attn_output, (-1, self.num_heads, tgt_len, self.head_dim))
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))
        return self.out_proj(attn_output)


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


class TextClipEmbedding(tf.keras.Model):
    def __init__(self, max_length, embed_dim=768, vocab_size=49408, name=None, ckpt_path=None):
        tokens = tf.keras.layers.Input(shape=(max_length,), dtype="int32", name="tokens")
        positions = tf.keras.layers.Input(shape=(max_length,), dtype="int32", name="positions")
        clip_emb = CLIPEmbedding(vocab_size, embed_dim, max_length, name="embeddings")([tokens, positions])
        super().__init__([tokens, positions], clip_emb, name=name)
        origin = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors"
        ckpt_mapping = [('text_model.embeddings.token_embedding.weight', None),
                        ('text_model.embeddings.position_embedding.weight', None)]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin, fname="text_encoder.safetensors")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping)


class TextEncoder(tf.keras.Model):
    def __init__(self, max_length=77, embed_dim=768, num_heads=12, num_layers=12, clip_skip=-2, name=None,
                 ckpt_path=None, lora_dict=None):
        clip_emb = tf.keras.layers.Input(shape=(max_length, embed_dim), dtype="float32", name="clip_emb")
        x = clip_emb
        out = []
        for idx in range(num_layers):
            x = CLIPEncoderLayer(embed_dim, num_heads, activation=quick_gelu,
                                 name="text_model.encoder.layers.{}".format(idx))(x)
            out.append(x)
        embedded = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="text_model.final_layer_norm")(out[clip_skip])
        super().__init__(clip_emb, embedded, name=name)
        origin = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.safetensors"
        ckpt_mapping = []
        for idx in range(0, num_layers + clip_skip + 1):
            layers_name = 'text_model.encoder.layers.{}'.format(idx)
            ckpt_mapping.append(('{}.layer_norm1.weight'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm1.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.q_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.q_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.k_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.k_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.v_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.v_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.self_attn.out_proj.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.self_attn.out_proj.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm2.weight'.format(layers_name), None))
            ckpt_mapping.append(('{}.layer_norm2.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.mlp.fc1.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.mlp.fc1.bias'.format(layers_name), None))
            ckpt_mapping.append(('{}.mlp.fc2.weight'.format(layers_name), (1, 0)))
            ckpt_mapping.append(('{}.mlp.fc2.bias'.format(layers_name), None))
        ckpt_mapping.append(('text_model.final_layer_norm.weight', None))
        ckpt_mapping.append(('text_model.final_layer_norm.bias', None))
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, lora_dict=lora_dict)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = tf.keras.utils.get_file(origin=origin, fname="text_encoder.safetensors")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, lora_dict=lora_dict)
