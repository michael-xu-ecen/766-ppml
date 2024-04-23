import torch
import torchio as tio
import pytorch_lightning as pl
class LightningModel(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
        #return batch['image'][batch_idx], batch['label'][batch_idx]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return x, y_hat, y

    def training_step(self, batch, batch_idx):
        x, y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def compute_training_step(self, batch, batch_idx):
        x, y, y_hat = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        return {
            "loss": loss,
            "transformed_batch": (x, y),
            "model_outputs": y_hat,
        }

    def validation_step(self, batch, batch_idx):
        x, y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def get_batch_gradients(
        self,
        batch: torch.tensor,
        batch_idx: int = 0,
        *args,
    ):
        #self.train()
        self.zero_grad()
        training_step_results = self.compute_training_step(
            batch, batch_idx)

        batch_gradients = torch.autograd.grad(
            training_step_results['loss'],
            self.net.parameters(),
        )
        return batch_gradients, training_step_results