# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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

from typing import Optional

import numpy as np


class Scheduler(object):
    """
    `LCMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.


    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        active_lcm (`bool`, defaults true):
            apply lcm or not.
        original_inference_steps (`int`, *optional*, defaults to 50):
            The default number of inference steps used to generate a linearly-spaced timestep schedule, from which we
            will ultimately take `num_inference_steps` evenly spaced timesteps to form the final timestep schedule.
        timestep_scaling (`float`, defaults to 10.0):
            The factor the timesteps will be multiplied by when calculating the consistency model boundary conditions
            `c_skip` and `c_out`. Increasing this will decrease the approximation error (although the approximation
            error at the default of `10.0` is already pretty small).
    """

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012,
                 original_inference_steps: int = 50, timestep_scaling: float = 10.0, active_lcm=True):
        self.active_lcm = active_lcm
        self.num_train_timesteps = num_train_timesteps
        self.original_inference_steps = original_inference_steps
        self.timestep_scaling = timestep_scaling
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
        self._step_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        index_candidates = np.nonzero(self.timesteps == timestep)
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(index_candidates) > 1:
            step_index = index_candidates[1]
        else:
            step_index = index_candidates[0]
        self._step_index = step_index

    @property
    def step_index(self):
        return self._step_index

    def set_timesteps(self, num_inference_steps: int, original_inference_steps: Optional[int] = None,
                      strength: int = 1.0):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
        """

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps.")
        self.num_inference_steps = num_inference_steps
        if self.active_lcm:
            original_steps = (
                original_inference_steps if original_inference_steps is not None else self.original_inference_steps)

            if original_steps > self.num_train_timesteps:
                raise ValueError(
                    f"`original_steps`: {original_steps} cannot be larger than `self.config_train_timesteps`:"
                    f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.num_train_timesteps} timesteps.")
            if num_inference_steps > original_steps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
                    f" {original_steps} because the final timestep schedule will be a subset of the"
                    f" `original_inference_steps`-sized initial timestep schedule.")
            # LCM Timesteps Setting
            # Currently, only linear spacing is supported.
            c = self.num_train_timesteps // original_steps
            # LCM Training Steps Schedule
            lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * c - 1
            skipping_step = len(lcm_origin_timesteps) // num_inference_steps
            # LCM Inference Steps Schedule
            timesteps = lcm_origin_timesteps[::-skipping_step][:num_inference_steps]
        else:
            timesteps = np.linspace(0, 1000 - 1, num_inference_steps, dtype=np.int32)[::-1]
        self.timesteps = timesteps.copy().astype(np.int32)
        self._step_index = None

    def get_scalings_for_boundary_condition_discrete(self, timestep, sigma_data=0.5):
        scaled_timestep = timestep * self.timestep_scaling
        c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
        c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5
        return c_skip, c_out

    def step(self, latent: np.ndarray, timestep: int, latent_prev: np.ndarray):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            latent (`np.ndarray`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            latent_prev (`np.ndarray`):
                A current instance of a sample created by the diffusion process.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)
        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep
        next_signal_rates = self.signal_rates[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        next_noise_rates = self.noise_rates[prev_timestep]
        signal_rates = self.signal_rates[timestep]
        noise_rates = self.noise_rates[timestep]
        # 2. Compute the predicted original sample x_0 based on the model parameterization
        pred_x0 = (latent_prev - noise_rates * latent) / signal_rates
        # 3. Denoise model output using boundary conditions
        if self.active_lcm:
            # 4. Get scalings for boundary conditions
            c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
            denoised = c_out * pred_x0 + c_skip * latent_prev
            # 5. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            # Noise is not used on the final timestep of the timestep schedule.
            # This also means that noise is not used for one-step sampling.
            if self.step_index != self.num_inference_steps - 1:
                noise = np.random.randn(*latent.shape).astype(np.float32)
                latent = next_signal_rates * denoised + next_noise_rates * noise
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
