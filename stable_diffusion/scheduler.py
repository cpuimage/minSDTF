# Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion
from typing import List, Optional

import numpy as np


class Scheduler(object):
    """
    `Scheduler` incorporates the `Strategic Stochastic Sampling` introduced by the paper `Trajectory Consistency
    Distillation`, extending the original Multistep Consistency Sampling to enable unrestricted trajectory traversal.

    This code is based on the official repo of TCD(https://github.com/jabir-zheng/TCD).


    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        active_tcd (`bool`, defaults true):
            apply tcd or not.
        original_inference_steps (`int`, *optional*, defaults to 50):
            The default number of inference steps used to generate a linearly-spaced timestep schedule, from which we
            will ultimately take `num_inference_steps` evenly spaced timesteps to form the final timestep schedule.
    """

    order = 1

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012,
                 original_inference_steps: int = 50, active_tcd: bool = True):
        self.active_tcd = active_tcd
        self.num_train_timesteps = num_train_timesteps
        self.original_inference_steps = original_inference_steps
        # this schedule is very specific to the latent diffusion model.
        self.alphas_cumprod = np.cumprod(
            1. - np.square(np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), num_train_timesteps)), axis=0)
        self.signal_rates = np.sqrt(self.alphas_cumprod)
        self.noise_rates = np.sqrt(1. - self.alphas_cumprod)
        self.final_alpha_cumprod = 1.0

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int32)
        self.custom_timesteps = False

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = np.nonzero(schedule_timesteps == timestep)
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0
        return indices[pos]

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: Optional[int] = None,
                      original_inference_steps: Optional[int] = None, timesteps: Optional[List[int]] = None,
                      strength: float = 1.0):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
            strength (`float`, *optional*, defaults to 1.0):
                Used to determine the number of timesteps used for inference when using img2img, inpaint, etc.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")
        if self.active_tcd:
            # 1. Calculate the TCD original training/distillation timestep schedule.
            original_steps = (
                original_inference_steps if original_inference_steps is not None else self.original_inference_steps)
            if original_inference_steps is None:
                # default option, timesteps align with discrete inference steps
                if original_steps > self.num_train_timesteps:
                    raise ValueError(
                        f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                        f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                        f" maximal {self.num_train_timesteps} timesteps.")
                # TCD Timesteps Setting
                # The skipping step parameter k from the paper.
                k = self.num_train_timesteps // original_steps
                # TCD Training/Distillation Steps Schedule
                tcd_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1
            else:
                # customised option, sampled timesteps can be any arbitrary value
                tcd_origin_timesteps = np.asarray(list(range(0, int(self.num_train_timesteps * strength))))

            # 2. Calculate the TCD inference timestep schedule.
            if timesteps is not None:
                # 2.1 Handle custom timestep schedules.
                train_timesteps = set(tcd_origin_timesteps)
                non_train_timesteps = []
                for i in range(1, len(timesteps)):
                    if timesteps[i] >= timesteps[i - 1]:
                        raise ValueError("`custom_timesteps` must be in descending order.")

                    if timesteps[i] not in train_timesteps:
                        non_train_timesteps.append(timesteps[i])

                if timesteps[0] >= self.num_train_timesteps:
                    raise ValueError(
                        f"`timesteps` must start before `self.config.train_timesteps`:"
                        f" {self.num_train_timesteps}.")

                # Raise warning if timestep schedule does not start with self.config.num_train_timesteps - 1
                if strength == 1.0 and timesteps[0] != self.num_train_timesteps - 1:
                    print(
                        f"The first timestep on the custom timestep schedule is {timesteps[0]}, not"
                        f" `self.config.num_train_timesteps - 1`: {self.num_train_timesteps - 1}. You may get"
                        f" unexpected results when using this timestep schedule.")
                # Raise warning if custom timestep schedule contains timesteps not on original timestep schedule
                if non_train_timesteps:
                    print(
                        f"The custom timestep schedule contains the following timesteps which are not on the original"
                        f" training/distillation timestep schedule: {non_train_timesteps}. You may get unexpected results"
                        f" when using this timestep schedule.")
                # Raise warning if custom timestep schedule is longer than original_steps
                if original_steps is not None:
                    if len(timesteps) > original_steps:
                        print(
                            f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
                            f" the length of the timestep schedule used for training: {original_steps}. You may get some"
                            f" unexpected results when using this timestep schedule.")
                else:
                    if len(timesteps) > self.num_train_timesteps:
                        print(
                            f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
                            f" the length of the timestep schedule used for training: {self.num_train_timesteps}. You may get some"
                            f" unexpected results when using this timestep schedule.")
                timesteps = np.array(timesteps, dtype=np.int32)
                self.num_inference_steps = len(timesteps)
                self.custom_timesteps = True
                # Apply strength (e.g. for img2img pipelines) (see StableDiffusionImg2ImgPipeline.get_timesteps)
                init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
                t_start = max(self.num_inference_steps - init_timestep, 0)
                timesteps = timesteps[t_start * self.order:]
                # TODO: also reset self.num_inference_steps?
            else:
                # 2.2 Create the "standard" TCD inference timestep schedule.
                if num_inference_steps > self.num_train_timesteps:
                    raise ValueError(
                        f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                        f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                        f" maximal {self.num_train_timesteps} timesteps.")
                if original_steps is not None:
                    skipping_step = len(tcd_origin_timesteps) // num_inference_steps
                    if skipping_step < 1:
                        raise ValueError(
                            f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}.")
                self.num_inference_steps = num_inference_steps
                if original_steps is not None:
                    if num_inference_steps > original_steps:
                        raise ValueError(
                            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                            f" {original_steps} because the final timestep schedule will be a subset of the"
                            f" `original_inference_steps`-sized initial timestep schedule.")
                else:
                    if num_inference_steps > self.num_train_timesteps:
                        raise ValueError(
                            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:"
                            f" {self.num_train_timesteps} because the final timestep schedule will be a subset of the"
                            f" `num_train_timesteps`-sized initial timestep schedule.")

                # TCD Inference Steps Schedule
                tcd_origin_timesteps = tcd_origin_timesteps[::-1].copy()
                # Select (approximately) evenly spaced indices from tcd_origin_timesteps.
                inference_indices = np.linspace(0, len(tcd_origin_timesteps), num=num_inference_steps, endpoint=False)
                inference_indices = np.floor(inference_indices).astype(np.int32)
                timesteps = tcd_origin_timesteps[inference_indices]
        else:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(0, 1000, num_inference_steps, dtype=np.int32, endpoint=False)
            timesteps = timesteps[::-1]
        self.timesteps = timesteps.copy().astype(np.int32)
        self._step_index = None
        self._begin_index = None

    def step(self, latent: np.ndarray, timestep: int, latent_prev: np.ndarray, eta: float = 0.3):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            latent (`np.ndarray`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            latent_prev (`np.ndarray`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                A stochastic parameter (referred to as `gamma` in the paper) used to control the stochasticity in every
                step. When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic
                sampling.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        assert 0 <= eta <= 1.0, "gamma must be less than or equal to 1.0"

        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = 0 if self.active_tcd else timestep

        # 2. compute alphas, betas
        next_signal_rates = self.signal_rates[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        next_noise_rates = self.noise_rates[prev_timestep]
        signal_rates = self.signal_rates[timestep]
        noise_rates = self.noise_rates[timestep]
        # 3. Compute the predicted original sample x_0 based on the model parameterization
        pred_x0 = (latent_prev - noise_rates * latent) / signal_rates
        if self.active_tcd:
            timestep_s = np.floor((1. - eta) * prev_timestep).astype(np.int32)
            alpha_from_s = self.alphas_cumprod[timestep_s]
            beta_prod_s = 1. - alpha_from_s
            noise_rates_s = np.sqrt(beta_prod_s)
            signal_rates_s = np.sqrt(alpha_from_s)
            denoised = signal_rates_s * pred_x0 + noise_rates_s * latent
            # 4. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            # Noise is not used on the final timestep of the timestep schedule.
            # This also means that noise is not used for one-step sampling.
            # Eta (referred to as "gamma" in the paper) was introduced to control the stochasticity in every step.
            # When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic sampling.
            if eta > 0.0:
                alphas_to = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
                if self.step_index != self.num_inference_steps - 1:
                    noise = np.random.randn(*latent.shape).astype(np.float32)
                    latent = np.sqrt(alphas_to / alpha_from_s) * denoised + np.sqrt(
                        1.0 - alphas_to / alpha_from_s) * noise
                else:
                    latent = denoised
            else:
                latent = denoised
        else:
            if self.step_index != self.num_inference_steps - 1:
                latent = next_signal_rates * pred_x0 + next_noise_rates * latent
            else:
                latent = pred_x0
        # upon completion increase step index by one
        self._step_index += 1
        return latent

    def __len__(self):
        return self.num_train_timesteps
