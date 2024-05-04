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
        #self.grad_prune = False
        #self.prune_ratio = 0.9

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.net.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return x, y_hat, y

    def training_step(self, batch, batch_idx):
        x, y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        # loss.backward(retain_graph=True)
        # if self.grad_prune:
        #     print("hello")
        #     input_grads = [p.grad for p in self.net.parameters()]
        #     threshold = [
        #         torch.quantile(torch.abs(input_grads[i]), self.prune_ratio)
        #         for i in range(len(input_grads))
        #     ]
        #     for i, p in enumerate(self.net.parameters()):
        #         idx = torch.abs(p.grad) < threshold[i]
        #         p.grad[idx] = 0
        return loss


    def validation_step(self, batch, batch_idx):
        x, y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
