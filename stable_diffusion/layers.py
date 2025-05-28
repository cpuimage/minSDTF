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
from keras import layers, activations, ops


class PaddedConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = layers.ZeroPadding2D(padding)
        self.conv2d = layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)


class AttentionBlock(layers.Layer):
    def __init__(self, output_dim, q_chunk_size=16, kv_chunk_size=1, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.GroupNormalization(epsilon=1e-5)
        self.to_q = layers.Dense(output_dim, use_bias=True, )
        self.to_k = layers.Dense(output_dim, use_bias=True, )
        self.to_v = layers.Dense(output_dim, use_bias=True, )
        self.out_proj = layers.Dense(output_dim, use_bias=True, )
        self.scale = output_dim ** -0.5
        self.output_dim = output_dim
        self.q_chunk_size = q_chunk_size
        self.kv_chunk_size = kv_chunk_size

    @staticmethod
    def _flash_attention(query, key, value, scale, q_chunk_size=1, kv_chunk_size=1):
        # FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
        # https://arxiv.org/abs/2205.14135v2
        # Self-attention Does Not Need O(n2) Memory
        # https://arxiv.org/abs/2112.05682v2
        result = []
        if kv_chunk_size == 1:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                q = q_chunks[q_chunk_idx]
                score = ops.einsum('bqh,bkh->bqk', q, key) * scale
                context = ops.softmax(score)
                acc = ops.einsum('bqk,bkh->bqh', context, value)
                result.append(acc)
        else:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            k_chunks = ops.split(key, kv_chunk_size, axis=1)
            v_chunks = ops.split(value, kv_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                global_max = 0.0 - float("inf")
                global_max_diffs = 0.0
                acc = 0.0
                q = q_chunks[q_chunk_idx]
                for kv_chunk_idx in range(kv_chunk_size):
                    k = k_chunks[kv_chunk_idx]
                    v = v_chunks[kv_chunk_idx]
                    attn_weights = ops.einsum('bqh,bkh->bqk', q, k) * scale
                    curr_max_score = ops.maximum(ops.max(attn_weights, axis=-1, keepdims=True), global_max)
                    global_max_diffs = global_max_diffs * ops.exp(global_max - curr_max_score)
                    exp_weights = ops.exp(attn_weights - curr_max_score)
                    curr_max_diffs = ops.sum(exp_weights, axis=-1, keepdims=True) + global_max_diffs
                    acc = (acc * global_max_diffs + ops.einsum('bqk,bkh->bqh', exp_weights, v)) / curr_max_diffs
                    global_max_diffs = curr_max_diffs
                    global_max = curr_max_score
                result.append(acc)
        return ops.concatenate(result, axis=1)

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        x = self.norm(inputs)
        shape = ops.shape(x)
        h, w, c = shape[1], shape[2], shape[3]
        x = ops.reshape(x, (-1, h * w, c))  # b, hw, c
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out_shape = (batch_size, h, w, self.output_dim)
        if self.q_chunk_size == 1 and self.kv_chunk_size == 1:
            score = ops.einsum('bqh,bkh->bqk', q, k) * self.scale
            context = ops.softmax(score)
            attn = ops.einsum('bqk,bkh->bqh', context, v)
        else:
            attn = self._flash_attention(q, k, v, self.scale, q_chunk_size=self.q_chunk_size,
                                         kv_chunk_size=self.kv_chunk_size)
        out = ops.reshape(attn, out_shape)
        return self.out_proj(out) + inputs


class ResnetBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(activations.silu(self.norm1(inputs)))
        x = self.conv2(activations.silu(self.norm2(x)))
        return x + self.residual_projection(inputs)
