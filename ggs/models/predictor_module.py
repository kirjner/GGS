from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric, SpearmanCorrCoef, PearsonCorrCoef, MeanAbsoluteError
from ggs.models.predictors import BaseCNN


class PredictorModule(LightningModule):
    """

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_o ptimizers)
    """

    def __init__(self, model_cfg):
        super().__init__()
        self._cfg = model_cfg

        self.predictor = BaseCNN(
            **self._cfg.predictor,
        )
        self.optimizer = torch.optim.Adam(
            params=self.predictor.parameters(),
            **self._cfg.optimizer,
        )
        self.min_fluorescence = 0

        #loss function
        self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.L1Loss()
        self.train_sr = SpearmanCorrCoef()
        self.val_sr = SpearmanCorrCoef()
        self.test_sr = SpearmanCorrCoef()

        # self.train_pr = PearsonCorrCoef()
        # self.val_pr = PearsonCorrCoef()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_sr_best = MaxMetric()
        self.val_loss_best = MinMetric()
        self.val_pr_best = MaxMetric()
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.predictor(x)

    # def on_train_start(self):
    #     self.val_loss.reset()
    #     self.val_sr.reset()
    #     #self.val_pr.reset()
    #     self.val_sr_best.reset()
    #     #self.val_pr_best.reset()
    #     self.val_loss_best.reset()

    def model_step(self, batch: Any):
        xs, targets = batch
        targets = targets.float()
        preds = self.forward(xs)
        loss = self.criterion(targets, preds)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss(loss)
        #only report spearman/pearsonr on values with ground truth above the min_fluorescence, supports negative data augmentation
        non_augmented = targets > self.min_fluorescence
        if torch.sum(non_augmented) > 0:
            preds = preds[non_augmented]
            targets = targets[non_augmented]

        # update and log metrics
        self.train_sr(preds, targets)
        self.log("train/spearmanr", self.train_sr, on_step=False, on_epoch=True, prog_bar=True)
        #self.train_pr(preds, targets)
        #self.log("train/pearsonr", self.train_pr, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        # log metrics at the end of the epoch
        self.log("train/loss_epoch", self.train_loss.compute(), prog_bar=True)
        self.log("train/spearmanr_epoch", self.train_sr.compute(), prog_bar=True)


    '''
    A validation step can be implemented easily using the commented out code below
    '''
    # def validation_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.model_step(batch)
    #     self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.val_loss(loss)
    #     #only report spearman/pearsonr on values with ground truth above the min_fluorescence, supports negative data augmentation
    #     non_augmented = targets > self.min_fluorescence
    #     if torch.sum(non_augmented) > 0:
    #         preds = preds[non_augmented]
    #         targets = targets[non_augmented]

    #     # update and log metrics
    #     self.val_sr(preds, targets)
    #     self.val_pr(preds, targets)
    #     self.val_mae(preds, targets)
    #     self.log("val/spearmanr", self.val_sr, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("val/pearsonr", self.val_pr, on_step=False, on_epoch=True, prog_bar = True)
    #     self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     sr = self.val_sr.compute()
    #     pr = self.val_pr.compute()
    #     mae = self.val_mae.compute()
    #     loss = self.val_loss.compute()
    #     self.val_sr_best(sr)
    #     self.val_pr_best(pr)
    #     self.val_mae_best(mae)
    #     self.val_loss_best(loss)
    #     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    #     # otherwise metric would be reset by lightning after each epoch
    #     self.log("val/spearmanr_best", self.val_sr_best.compute(), prog_bar=True)
    #     self.log("val/pearsonr_best", self.val_pr_best.compute(), prog_bar=True)
    #     self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
    #     self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_loss(loss)
        #only report spearman/pearsonr on values with ground truth above the min_fluorescence, supports negative data augmentation
        non_augmented = targets > self.min_fluorescence
        if torch.sum(non_augmented) > 0:
            preds = preds[non_augmented]
            targets = targets[non_augmented]

        # update and log metrics
        self.test_sr(preds, targets)
        self.test_mae(preds, targets)
        self.log("test/spearmanr", self.test_sr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/spearmanr", self.test_sr.compute(), prog_bar=True)
        self.log("test/mae_best", self.test_mae.compute(), prog_bar=True)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}