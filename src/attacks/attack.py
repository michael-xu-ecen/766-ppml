# plugs attacks into existing gradient inversion attacks framework
import copy
from typing import Callable

import numpy as np
import torch
import pytorch_lightning as pl

from TrainingPipeline import TrainingPipeline
from inversefed import GradientReconstructor


class GradientInversionAttack:
    """Wrapper around Gradient Inversion attacks"""
    def __init__(
        self,
        pipeline: TrainingPipeline,
        dm,
        ds,
        device: torch.device,
        loss_metric: Callable,
        reconstructor_args: dict = None,
    ):
        self.pipeline = pipeline
        self._model = pipeline.model
        self.device = device
        self.loss_metric = loss_metric
        if reconstructor_args is None:
            reconstructor_args = {}
        self.reconstructor: GradientReconstructor = GradientReconstructor(
            self._model, mean_std=(dm, ds), **reconstructor_args)
        self.grad_prune = False
        self.grad_noise = False
        self.grad_clip = False
        self.keep_pruned = True

    def run_attack_batch(self, batch, grad_prune=False, grad_noise=False, grad_clip=False, keep_pruned = False, vanilla = False):
        """Runs an attacks given a batch of inputs and targets. Both should be tensors of shape (N, ...), where N
        is the number of inputs in the batch"""
        self._model.to(self.device)
        #self._model.zero_grad()
        self._model.train()
        inputs, targets = self._model.prepare_batch(batch)
        output = self._model.net(inputs.to(self._model.device))
        loss = self.loss_metric(output, targets)
        batch_gradients = torch.autograd.grad(loss,
                                              self._model.parameters(),
                                              create_graph=False)
        ## Simulated defenses
        if not vanilla:
            if grad_prune:
                threshold = [
                    torch.quantile(torch.abs(batch_gradients[i]), 0.8)
                    for i in range(len(batch_gradients))
                ]

            for i in range(len(batch_gradients)):
                try:
                    for j in range(len(batch_gradients[i])):
                        for k in range(len(batch_gradients[i][j])):
                            for l in range(len(batch_gradients[i][j][k])):
                                for m in range(len(batch_gradients[i][j][k][l])):
                                    for n in range(len(batch_gradients[i][j][k][l][m])):
                                        batch_gradient = batch_gradients[i][j][k][l][m][n]
                                        if grad_prune:
                                            if batch_gradient < threshold[i]:
                                                batch_gradient = 0
                                        if grad_clip:
                                            max_norm = .5
                                            batch_gradient = batch_gradient / max(
                                                1, torch.norm(batch_gradient, p=2) / max_norm)
                                        if grad_noise:
                                            if keep_pruned:
                                                if batch_gradient == 0:
                                                    continue
                                                else:
                                                    noise = np.random.normal(0, 1)
                                                    batch_gradient += noise
                except:
                    continue
        return self.run_attack_gradient(batch_gradients,
                                        input_shape=inputs[0].shape,
                                        labels=targets)

    def run_attack_gradient(self,
                            batch_gradients: torch.tensor,
                            input_shape: tuple,
                            labels=None):
        return self.reconstructor.reconstruct(batch_gradients,
                                              img_shape=input_shape,
                                              labels=labels)

    def run_from_dump(self, filepath: str, dataset: torch.utils.data.Dataset):
        loaded_data = torch.load(filepath, map_location=self.device)
        self._model.load_state_dict(loaded_data["model_state_dict"])
        batch_inputs, batch_targets = (
            dataset[i][0] for i in loaded_data["batch_indices"]), (
                dataset[i][1] for i in loaded_data["batch_indices"])
        return self.run_attack_batch(batch_inputs, batch_targets)