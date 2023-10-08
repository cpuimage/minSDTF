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
import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization.

    Args:
      groups: Integer, the number of groups for Group Normalization. Can be in
        the range [1, N] where N is the input dimension. The input dimension
        must be divisible by the number of groups. Defaults to 32.
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. This argument defaults to `-1`, the last
        dimension in the input.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to True.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
      beta_regularizer: Optional regularizer for the beta weight. None by
        default.
      gamma_regularizer: Optional regularizer for the gamma weight. None by
        default.
      beta_constraint: Optional constraint for the beta weight. None by default.
      gamma_constraint: Optional constraint for the gamma weight. None by
        default.  Input shape: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis) when using this
        layer as the first layer in a model.  Output shape: Same shape as input.
    Reference: - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
            self,
            groups=32,
            axis=-1,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def validate_axis(self, axis, input_shape):
        """Validate an axis value and returns its standardized form.

        Args:
          axis: Value to validate. Can be an integer or a list/tuple of integers.
            Integers may be negative.
          input_shape: Reference input shape that the axis/axes refer to.

        Returns:
          Normalized form of `axis`, i.e. a list with all-positive values.
        """
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank
        if not rank:
            raise ValueError(
                f"Input has undefined rank. Received: input_shape={input_shape}")
        # Convert axis to list and resolve negatives
        if isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)
        for idx, x in enumerate(axis):
            if x < 0:
                axis[idx] = rank + x
        # Validate axes
        for x in axis:
            if x < 0 or x >= rank:
                raise ValueError(
                    "Invalid value for `axis` argument. "
                    "Expected 0 <= axis < inputs.rank (with "
                    f"inputs.rank={rank}). Received: axis={tuple(axis)}")
        if len(axis) != len(set(axis)):
            raise ValueError(f"Duplicate axis: {tuple(axis)}")
        return axis

    def build(self, input_shape):
        self.validate_axis(self.axis, input_shape)
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}.")

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim}).")

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim}).")

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim})

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint, )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint, )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(inputs)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, input_shape)
        return tf.reshape(normalized_inputs, input_shape)

    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[i] for i in range(inputs.shape.rank)]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_reduction_axes = list(range(1, reshaped_inputs.shape.rank))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True)
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon, )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * tf.keras.backend.int_shape(input_shape)[0]
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


class PaddedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.padding2d = tf.keras.layers.ZeroPadding2D(padding)
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = GroupNormalization(epsilon=1e-5)
        self.q = tf.keras.layers.Dense(output_dim, use_bias=True, )
        self.k = tf.keras.layers.Dense(output_dim, use_bias=True, )
        self.v = tf.keras.layers.Dense(output_dim, use_bias=True, )
        self.proj_out = tf.keras.layers.Dense(output_dim, use_bias=True, )

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = tf.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = tf.reshape(q, (-1, h * w, c))  # b, hw, c
        k = tf.transpose(k, (0, 3, 1, 2))
        k = tf.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / tf.sqrt(tf.cast(c, self.compute_dtype))
        y = tf.keras.activations.softmax(y)

        # Attend to values
        v = tf.transpose(v, (0, 3, 1, 2))
        v = tf.reshape(v, (-1, c, h * w))
        y = tf.transpose(y, (0, 2, 1))
        x = v @ y
        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm1 = GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(output_dim, 3, padding=1)
        self.norm2 = GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(output_dim, 3, padding=1)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        x = self.conv1(tf.keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(tf.keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)
