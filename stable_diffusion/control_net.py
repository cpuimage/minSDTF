import os

from keras import layers, Model, Sequential, utils

from .ckpt_loader import load_weights_from_file, CKPT_MAPPING
from .diffusion_model import Attentions, ResBlock
from .layers import PaddedConv2D


class HintNet(Sequential):
    def __init__(self, img_height=512, img_width=512, name=None, controlnet_path=None):
        super().__init__(
            [
                layers.Input((img_height, img_width, 3)),
                PaddedConv2D(16, kernel_size=3, padding=1),
                layers.Activation("swish"),
                PaddedConv2D(16, kernel_size=3, padding=1),
                layers.Activation("swish"),
                PaddedConv2D(32, kernel_size=3, padding=1, strides=2),
                layers.Activation("swish"),
                PaddedConv2D(32, kernel_size=3, padding=1),
                layers.Activation("swish"),
                PaddedConv2D(96, kernel_size=3, padding=1, strides=2),
                layers.Activation("swish"),
                PaddedConv2D(96, kernel_size=3, padding=1),
                layers.Activation("swish"),
                PaddedConv2D(256, kernel_size=3, padding=1, strides=2),
                layers.Activation("swish"),
                PaddedConv2D(320, kernel_size=3, padding=1),
            ],
            name=name)
        origin = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth"
        ckpt_mapping = CKPT_MAPPING["hintnet"]
        if controlnet_path is not None:
            if os.path.exists(controlnet_path):
                load_weights_from_file(self, controlnet_path, ckpt_mapping=ckpt_mapping)
                return
            else:
                origin = controlnet_path
        model_weights_fpath = utils.get_file(origin=origin, fname="control_sd15_canny.pth")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping)


class ControlNet(Model):
    def __init__(self, img_height=512, img_width=512, name=None, controlnet_path=None):
        context = layers.Input((None, 768))
        t_embed_input = layers.Input((320,))
        latent = layers.Input((img_height // 8, img_width // 8, 4))
        hint_out = layers.Input((img_height // 8, img_width // 8, 320))
        t_emb = layers.Dense(1280, name="time_embedding.linear_1")(t_embed_input)
        t_emb = layers.Activation("swish")(t_emb)
        t_emb = layers.Dense(1280, name="time_embedding.linear_2",
                             activation=layers.Activation("swish"))(t_emb)
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent) + hint_out
        outputs.append(x)
        x = ResBlock(320)([x, t_emb])
        x = Attentions(8, 40, fully_connected=False)([x, context])
        outputs.append(x)
        x = ResBlock(320)([x, t_emb])
        x = Attentions(8, 40, fully_connected=False)([x, context])
        outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)
        outputs.append(x)
        x = ResBlock(640)([x, t_emb])
        x = Attentions(8, 80, fully_connected=False)([x, context])
        outputs.append(x)
        x = ResBlock(640)([x, t_emb])
        x = Attentions(8, 80, fully_connected=False)([x, context])
        outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)
        outputs.append(x)
        x = ResBlock(1280)([x, t_emb])
        x = Attentions(8, 160, fully_connected=False)([x, context])
        outputs.append(x)
        x = ResBlock(1280)([x, t_emb])
        x = Attentions(8, 160, fully_connected=False)([x, context])
        outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)
        outputs.append(x)
        x = ResBlock(1280)([x, t_emb])
        outputs.append(x)
        x = ResBlock(1280)([x, t_emb])
        outputs.append(x)
        x = ResBlock(1280)([x, t_emb])
        x = Attentions(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])
        outputs.append(x)
        assert len(outputs) == 13
        outs = []
        zero_convs = [PaddedConv2D(320, kernel_size=1, padding=0),
                      PaddedConv2D(320, kernel_size=1, padding=0),
                      PaddedConv2D(320, kernel_size=1, padding=0),
                      PaddedConv2D(320, kernel_size=1, padding=0),
                      PaddedConv2D(640, kernel_size=1, padding=0),
                      PaddedConv2D(640, kernel_size=1, padding=0),
                      PaddedConv2D(640, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0),
                      PaddedConv2D(1280, kernel_size=1, padding=0)]
        for i, x in enumerate(outputs):
            outs.append(zero_convs[i](x))
        super().__init__([latent, t_embed_input, context, hint_out], outs, name=name)
        origin = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth"
        ckpt_mapping = CKPT_MAPPING["controlnet"]
        if controlnet_path is not None:
            if os.path.exists(controlnet_path):
                load_weights_from_file(self, controlnet_path, ckpt_mapping=ckpt_mapping)
                return
            else:
                origin = controlnet_path
        model_weights_fpath = utils.get_file(origin=origin, fname="control_sd15_canny.pth")
        if os.path.exists(model_weights_fpath):
            load_weights_from_file(self, model_weights_fpath, ckpt_mapping=ckpt_mapping)
