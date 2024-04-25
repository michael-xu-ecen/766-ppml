# plugs attacks into existing gradient inversion attacks framework
import copy
from typing import Callable
import torch
import pytorch_lightning as pl
from inversefed import GradientReconstructor


class GradientInversionAttack:
    """Wrapper around Gradient Inversion attacks"""
    def __init__(
        self,
        model: pl.LightningModule,
        dm,
        ds,
        device: torch.device,
        loss_metric: Callable,
        reconstructor_args: dict = None,
    ):
        self._model = model
        self.device = device
        self.loss_metric = loss_metric
        if reconstructor_args is None:
            reconstructor_args = {}
        self.reconstructor: GradientReconstructor = GradientReconstructor(
            self._model, mean_std=(dm, ds), **reconstructor_args)

    def run_attack_batch(self, batch_inputs: torch.tensor,
                         batch_targets: torch.tensor):
        """Runs an attacks given a batch of inputs and targets. Both should be tensors of shape (N, ...), where N
        is the number of inputs in the batch"""
        self._model.to(self.device)
        self._model.zero_grad()
        self._model.train()
        output = self._model.net(batch_inputs.to(self._model.device))
        loss = self.loss_metric(output, batch_targets)
        batch_gradients = torch.autograd.grad(loss,
                                              self._model.parameters(),
                                              create_graph=False)
        return self.run_attack_gradient(batch_gradients,
                                        input_shape=batch_inputs[0].shape,
                                        labels=batch_targets)

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