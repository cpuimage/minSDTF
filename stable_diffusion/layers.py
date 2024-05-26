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
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = layers.GroupNormalization(epsilon=1e-5)
        self.q = layers.Dense(output_dim, use_bias=True, )
        self.k = layers.Dense(output_dim, use_bias=True, )
        self.v = layers.Dense(output_dim, use_bias=True, )
        self.proj_out = layers.Dense(output_dim, use_bias=True, )

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw, c
        k = ops.transpose(k, (0, 3, 1, 2))
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype))
        y = activations.softmax(y)

        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2))
        v = ops.reshape(v, (-1, c, h * w))
        y = ops.transpose(y, (0, 2, 1))
        x = v @ y
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs


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
