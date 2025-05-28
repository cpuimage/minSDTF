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

from keras import layers, Model, utils, activations, ops

from .ckpt_loader import load_weights_from_file, CKPT_MAPPING, UNET_KEY_MAPPING
from .layers import PaddedConv2D


class ResBlock(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            layers.GroupNormalization(epsilon=1e-5, name="norm1"),
            layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1, name="conv1")]
        self.embedding_flow = layers.Dense(output_dim, name="time_emb_proj")
        self.exit_flow = [
            layers.GroupNormalization(epsilon=1e-5, name="norm2"),
            layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1, name="conv2")]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1, name="conv_shortcut")
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        embeddings = self.embedding_flow(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class Attentions(layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.GroupNormalization(epsilon=1e-5, name="norm")
        channels = num_heads * head_size
        if fully_connected:
            self.proj_in = layers.Dense(num_heads * head_size, name="proj_in")
        else:
            self.proj_in = PaddedConv2D(num_heads * head_size, 1, name="proj_in")
        self.transformer_block = TransformerBlock(channels, num_heads, head_size, name="transformer_blocks.0")
        if fully_connected:
            self.proj_out = layers.Dense(channels, name="proj_out")
        else:
            self.proj_out = PaddedConv2D(channels, 1, name="proj_out")

    def call(self, inputs):
        inputs, context = inputs
        batch_size = ops.shape(inputs)[0]
        h, w, c = inputs.get_shape().as_list()[1:]
        x = self.norm(inputs)
        x = self.proj_in(x)
        x = ops.reshape(x, (batch_size, h * w, c))
        x = self.transformer_block([x, context])
        x = ops.reshape(x, (batch_size, h, w, c))
        return self.proj_out(x) + inputs


class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn1 = CrossAttention(num_heads, head_size, q_chunk_size=16, kv_chunk_size=1, name="attn1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.attn2 = CrossAttention(num_heads, head_size, q_chunk_size=16, kv_chunk_size=1, name="attn2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-5, name="norm3")
        self.geglu = GEGLU(dim * 4, name="ff.net.0")
        self.dense = layers.Dense(dim, name="ff.net.2")

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1(self.norm1(inputs), context=None) + inputs
        x = self.attn2(self.norm2(x), context=context) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class CrossAttention(layers.Layer):
    def __init__(self, num_heads, head_size, q_chunk_size=16, kv_chunk_size=1, **kwargs):
        super().__init__(**kwargs)
        self.to_q = layers.Dense(num_heads * head_size, use_bias=False, name="to_q")
        self.to_k = layers.Dense(num_heads * head_size, use_bias=False, name="to_k")
        self.to_v = layers.Dense(num_heads * head_size, use_bias=False, name="to_v")
        self.scale = head_size ** -0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = layers.Dense(num_heads * head_size, name="to_out")
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

    def call(self, inputs, context=None):
        context = inputs if context is None else context
        batch_size = ops.shape(inputs)[0]
        q_time = ops.shape(inputs)[1]
        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)
        out_shape = (batch_size, q_time, self.num_heads * self.head_size)
        num_heads_axis = -2
        v_out = []
        q_vec = ops.split(q, self.num_heads, axis=-1)
        k_vec = ops.split(k, self.num_heads, axis=-1)
        v_vec = ops.split(v, self.num_heads, axis=-1)
        for idx in range(self.num_heads):
            q = q_vec[idx]
            k = k_vec[idx]
            v = v_vec[idx]
            if self.q_chunk_size == 1 and self.kv_chunk_size == 1:
                score = ops.einsum('bqh,bkh->bqk', q, k) * self.scale
                context = ops.softmax(score)
                attn = ops.einsum('bqk,bkh->bqh', context, v)
            else:
                attn = self._flash_attention(q, k, v, self.scale, q_chunk_size=self.q_chunk_size,
                                             kv_chunk_size=self.kv_chunk_size)
            v_out.append(attn)
        attn = ops.stack(v_out, axis=num_heads_axis)
        out = ops.reshape(attn, out_shape)
        return self.out_proj(out)


class Upsamplers(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1, name="conv")

    def call(self, inputs):
        return self.conv(self.ups(inputs))


class GEGLU(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.proj = layers.Dense(output_dim * 2, name="proj")

    def call(self, inputs):
        x = self.proj(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim:]
        tanh_res = activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate ** 2)))
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    aa = ops.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = ops.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = layers.Dot(axes=(2, 1))([aa, bb])
    return ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


class DiffusionModel(Model):
    def __init__(self, img_height=512, img_width=512, apply_control_net=False, name=None,
                 ckpt_path=None, lora_dict=None):
        context = layers.Input((None, 768))
        t_embed_input = layers.Input((320,))
        latent = layers.Input((img_height // 8, img_width // 8, 4))
        controls = None
        if apply_control_net:
            controls = [layers.Input((img_height // 8, img_width // 8, 320)),
                        layers.Input((img_height // 8, img_width // 8, 320)),
                        layers.Input((img_height // 8, img_width // 8, 320)),
                        layers.Input((img_height // 16, img_width // 16, 320)),
                        layers.Input((img_height // 16, img_width // 16, 640)),
                        layers.Input((img_height // 16, img_width // 16, 640)),
                        layers.Input((img_height // 32, img_width // 32, 640)),
                        layers.Input((img_height // 32, img_width // 32, 1280)),
                        layers.Input((img_height // 32, img_width // 32, 1280)),
                        layers.Input((img_height // 64, img_width // 64, 1280)),
                        layers.Input((img_height // 64, img_width // 64, 1280)),
                        layers.Input((img_height // 64, img_width // 64, 1280)),
                        layers.Input((img_height // 64, img_width // 64, 1280))]
        t_emb = layers.Dense(1280, name="time_embedding.linear_1")(t_embed_input)
        t_emb = layers.Activation("swish")(t_emb)
        t_emb = layers.Dense(1280, name="time_embedding.linear_2",
                             activation=layers.Activation("swish"))(
            t_emb)
        # Downsampling flow
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1, name="conv_in")(latent)
        outputs.append(x)
        # down_blocks.0
        x = ResBlock(320, name="down_blocks.0.resnets.0")([x, t_emb])
        x = Attentions(8, 40, fully_connected=False, name="down_blocks.0.attentions.0")([x, context])
        outputs.append(x)
        x = ResBlock(320, name="down_blocks.0.resnets.1")([x, t_emb])
        x = Attentions(8, 40, fully_connected=False, name="down_blocks.0.attentions.1")([x, context])
        outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1, name="down_blocks.0.downsamplers.0")(x)  # Downsample 2x
        outputs.append(x)
        # down_blocks.1
        x = ResBlock(640, name="down_blocks.1.resnets.0")([x, t_emb])
        x = Attentions(8, 80, fully_connected=False, name="down_blocks.1.attentions.0")([x, context])
        outputs.append(x)
        x = ResBlock(640, name="down_blocks.1.resnets.1")([x, t_emb])
        x = Attentions(8, 80, fully_connected=False, name="down_blocks.1.attentions.1")([x, context])
        outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1, name="down_blocks.1.downsamplers.0")(x)  # Downsample 2x
        outputs.append(x)
        # down_blocks.2
        x = ResBlock(1280, name="down_blocks.2.resnets.0")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="down_blocks.2.attentions.0")([x, context])
        outputs.append(x)
        x = ResBlock(1280, name="down_blocks.2.resnets.1")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="down_blocks.2.attentions.1")([x, context])
        outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1, name="down_blocks.2.downsamplers.0")(x)  # Downsample 2x
        outputs.append(x)
        # down_blocks.3
        x = ResBlock(1280, name="down_blocks.3.resnets.0")([x, t_emb])
        outputs.append(x)
        x = ResBlock(1280, name="down_blocks.3.resnets.1")([x, t_emb])
        outputs.append(x)
        # mid_block
        # Middle flow
        x = ResBlock(1280, name="mid_block.resnets.0")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="mid_block.attentions.0")([x, context])
        x = ResBlock(1280, name="mid_block.resnets.1")([x, t_emb])
        if controls is not None:
            x = x + controls[12]
            assert len(outputs) == 12
            for i in range(len(outputs)):
                outputs[i] = outputs[i] + controls[i]
        # Upsampling flow
        # up_blocks.0
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.0.resnets.0")([x, t_emb])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.0.resnets.1")([x, t_emb])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.0.resnets.2")([x, t_emb])
        x = Upsamplers(1280, name="up_blocks.0.upsamplers.0")(x)
        # up_blocks.1
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.1.resnets.0")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="up_blocks.1.attentions.0")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.1.resnets.1")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="up_blocks.1.attentions.1")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(1280, name="up_blocks.1.resnets.2")([x, t_emb])
        x = Attentions(8, 160, fully_connected=False, name="up_blocks.1.attentions.2")([x, context])
        x = Upsamplers(1280, name="up_blocks.1.upsamplers.0")(x)
        # up_blocks.2
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(640, name="up_blocks.2.resnets.0")([x, t_emb])
        x = Attentions(8, 80, fully_connected=False, name="up_blocks.2.attentions.0")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(640, name="up_blocks.2.resnets.1")([x, t_emb])
        x = Attentions(8, 80, fully_connected=False, name="up_blocks.2.attentions.1")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(640, name="up_blocks.2.resnets.2")([x, t_emb])
        x = Attentions(8, 80, fully_connected=False, name="up_blocks.2.attentions.2")([x, context])
        x = Upsamplers(640, name="up_blocks.2.upsamplers.0")(x)
        # up_blocks.3
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(320, name="up_blocks.3.resnets.0")([x, t_emb])
        x = Attentions(8, 40, fully_connected=False, name="up_blocks.3.attentions.0")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(320, name="up_blocks.3.resnets.1")([x, t_emb])
        x = Attentions(8, 40, fully_connected=False, name="up_blocks.3.attentions.1")([x, context])
        x = layers.Concatenate(axis=-1)([x, outputs.pop()])
        x = ResBlock(320, name="up_blocks.3.resnets.2")([x, t_emb])
        x = Attentions(8, 40, fully_connected=False, name="up_blocks.3.attentions.2")([x, context])
        # Exit flow
        x = layers.GroupNormalization(epsilon=1e-5, name="conv_norm_out")(x)
        x = layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1, name="conv_out")(x)
        if controls is not None:
            super().__init__([latent, t_embed_input, context] + list(controls), output, name=name)
        else:
            super().__init__([latent, t_embed_input, context], output, name=name)
        origin = "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors"
        ckpt_mapping = CKPT_MAPPING["civitai_model"]
        if ckpt_path is not None:
            if os.path.exists(ckpt_path):
                load_weights_from_file(self, ckpt_path, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                       lora_dict=lora_dict)
                return
            else:
                origin = ckpt_path
        model_weights_fpath = utils.get_file(origin=origin, fname="dreamlike-photoreal-2.0.safetensors")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping, key_mapping=UNET_KEY_MAPPING,
                                   lora_dict=lora_dict)
