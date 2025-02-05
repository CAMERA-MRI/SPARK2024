from typing import Any, Dict, Tuple

import torch
import torchio as tio

from monai.metrics import CumulativeIterationMetric
from monai.losses import DiceLoss
# from monai.losses import DiceCELoss # Two changes in the file to incorporate DiceCELoss; first on this import and Second while instantiating DiceCELoss in the init. while instantiating added extra argument ce_weight for making it weighted dicece loss. Also, imported Parameter, created new class LearnableWeightedDiceCELoss
from torch.nn import Parameter

from monai.metrics.meandice import DiceMetric

import numpy as np
from lightning import LightningModule
# from lightning.pytorch.loggers import wandb # threw error in wandb.Image
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger


class BratsLitModule(LightningModule):
    """
    PyTorch Lightning module for training and evaluating models on the BRATS dataset.
    It implements 8 key methods of LightningModule
    """
    def __init__(self,
                 net: torch.nn.Module, # model to train
                 optimizer: torch.optim.Optimizer, # optimizer to use for training
                 scheduler: torch.optim.lr_scheduler, # learning rate scheduler to use for training
                #  compile: bool,
                 ) -> None:
        super().__init__()

        # save_hyperparameters() allows to access init params with 'self.hparams' attribute (i.e. saves hyperparameters including all attributes prefixed with 'self' of __init__ i.e. the arguments passed to __init__), also the hyperparameters saved by save_hyperparameters() are included in model checkpoints, regardless of the logger argument. logger=True ensures init params will also be logged by loggers.
        self.save_hyperparameters(logger=False)
        # self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # Initialize dice loss
        self.dice_loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        # self.dice_loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        
        # Initialize dice score for calculating validation dice score and test dice score
        self.dice_score_fn_train = DiceMetric(include_background=False)
        self.dice_score_fn_val = DiceMetric(include_background=False)
        self.dice_score_fn_test = DiceMetric(include_background=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model 'self.net'
        :param x: image of shape [BCDHW] -> [batch, 4, 155, 240, 240]
        :return: A tensor of logits of shape [BCDHW] -> [batch, 4, 155, 240, 240]
        """
        return self.net(X)

    def training_step(self,
                      batch,
                    #   batch: Dict[str, torch.Tensor ],
                      batch_idx: int):
        """Perform a single training step on a batch of data from training set.
        :param batch: A batch of data containing the input tensor of images and mask, accessed via key 'image' and 'mask'
        :param batch_idx: The index of the current batch
        :return: A tensor of losses between model predictions and targets
        """
        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        logits = self(X) # equivalent to self.forward(X)
        loss = self.dice_loss_fn(logits, Y) # logits of shape [batch, 4, 128, 240, 240] and loss_fn convertes Y to one-hot resulting the same shape as logits.
        loss.requires_grad_(True) # incase of transferring weights from pretrained checkpoint, need to set requires_grad=True for loss function.  # boolean attribute, not a method. so, requires_grad_(True), and not the requires_grad(True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # If logger=False, it (train_loss) won't be logged into csv file or TensorBoard depending upon logger=csv is used or something else.

        # Debugging
        # print(f"logits requires_grad: {logits.requires_grad}")
        # print(f"loss requires_grad: {loss.requires_grad}")

        # Log predicted_mask of train samples to wandb
        predicted_class_labels_train = torch.argmax(logits, dim=1, keepdim=True)
        self.dice_score_fn_train(predicted_class_labels_train, Y)
        self.log('train_score', self.dice_score_fn_train.aggregate().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.dice_score_fn_train.reset()

        if batch_idx == 1: # Log only for the first batch to reduce the number of images logged
            self.log_images(X, Y, predicted_class_labels_train, "train", batch_idx)
        return loss

        # PL automatically does gradient cleaning{model.zero_grad()}, accumulating derivatives{loss.backward()}, and step in opposite direction of gradient i.e. backpropagation (optimizer.step())
    
    def validation_step(self,
                        batch,
                        # batch: Dict[str, torch.Tensor],
                        batch_idx: int):
        """Perform a single validation step on a batch of data from the validation set"""

        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        val_logits = self(X)
        predicted_class_labels_val = torch.argmax(val_logits, dim=1, keepdim=True) # val_logits of shape [batch, 4, D, H, W] {raw logits}, after argmax [batch, D, H, W] {0, 1, 2, 3}, since it takes argmax along the channels (or, the #classes)
        sample_dice_list = self.dice_score_fn_val(predicted_class_labels_val, Y) # if batch_size is 2, then it gives 2 dice_scores. i.e. gives list of sample_wise dice_score here 2 samples so 2 dice_score, each averaged over the channels/classes.

        # If batch_size==1, then sample_dice_list is equal to val_avg_dice.
        val_avg_dice = self.dice_score_fn_val.aggregate().item() # gives batch dice_score; 2 dice_scores are aggregated into one 
        self.log('val_score', val_avg_dice, on_step=True, on_epoch=True, prog_bar=True, logger=True) # When on_epoch=True, the values logged at each step are accumulated, and their average is computed and logged at the end of the epoch by pytorch lightning itself.
        self.dice_score_fn_val.reset()

        # Log predicted mask of val samples to wandb
        if batch_idx == 0:
            self.log_images(X, Y, predicted_class_labels_val, 'val', batch_idx)
        # return sample_dice_list, val_avg_dice

    def test_step(self,
                  batch,
                #   batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set"""
        
        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        test_logits = self(X)
        predicted_class_labels_test = torch.argmax(test_logits, dim=1, keepdim=True)
        self.dice_score_fn_test(predicted_class_labels_test, Y)

        test_avg_dice = self.dice_score_fn_test.aggregate().item()
        self.log('test_score', test_avg_dice, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.dice_score_fn_test.reset()

        # Log predicted mask of test samples to wandb
        if batch_idx == 5:
            self.log_images(X, Y, predicted_class_labels_test, 'test', batch_idx)

    # def on_validation_epoch_end(self) -> None:
    #     """Lightning hook that is called when a validation epoch ends"""
    #     val_avg_dice = self.dice_score_fn_val.aggregate().item()
    #     self.log('val_score', val_avg_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.dice_score_fn_val.reset()

    # def on_test_epoch_end(self) -> None:
    #     """Lightning hook that is called when a test epoch ends"""
    #     test_avg_dice = self.dice_score_fn_test.aggregate().item()
    #     self.log('test_score', test_avg_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.dice_score_fn_test.reset()

    def configure_optimizers(self):
        """Configure what optimizers and learning-rate schedulers to use in our optimization"""
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters()) # Creates an optimizer using a class stored in self.hparams.optimizer
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_score",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def log_images(self, X, Y, predicted_class_labels, stage, batch_idx):
        # Convert tensors to numpy arrays for logging into wandb
        X = X.cpu().float().numpy() # [B, C=4, H, W, D]
        Y = Y.cpu().float().numpy() # [B, C=1, H, W, D]
        predicted = predicted_class_labels.cpu().float().numpy() # [B, C=1, H, W, D]

       # Take a middle slice from input (X), ground mask (Y), predicted mask (predicted)
        slice_idx = X.shape[4]//2 # middle_slcie along the depth
        middle_slice_input = X[0, 0, :, :, slice_idx] # C=0 is taken i.e. one of the four modalities
        middle_slice_ground = Y[0, 0, :, :, slice_idx]
        middle_slice_pred = predicted[0, 0, :, :, slice_idx]

        if isinstance(self.logger, WandbLogger):
            # Log the input, ground mask, and predicted mask
            self.logger.experiment.log({
                f"{stage}_ground_truth_{batch_idx}": wandb.Image(
                    self.apply_color_map(middle_slice_ground),
                    caption=f"{stage.capitalize()} Ground Mask, Epoch {self.current_epoch}, Batch {batch_idx}, Slice_idx {Y.shape[4]//2}"
                ),

                f"{stage}_predicted_mask_{batch_idx}": wandb.Image(
                    self.apply_color_map(middle_slice_pred), 
                    caption=f"{stage.capitalize()} Predicted Mask, Epoch {self.current_epoch}, Batch {batch_idx}, slice_idx{predicted.shape[4]//2}"
                ), 
                
                f"{stage}_image_{batch_idx}": wandb.Image(
                    middle_slice_input,
                    caption=f"{stage.capitalize()} Input, Epoch {self.current_epoch}, Batch {batch_idx}, Slice_idx {X.shape[4]//2}"
                ),

            })

    def apply_color_map(self, mask):
        # Define custom colors for each class
        colors = {
            0: [0, 0, 0],       # Black for background
            1: [255, 0, 0],     # Red for Non-Enhancing Tumor
            2: [255, 255, 0],   # Yellow for Enhancing Region
            3: [0, 255, 0],     # Green for Edema/Whole Tumor
        }

        unique_classes = np.unique(mask) # Identify all unique class values in the mask, [0., 1., 2., 3.]
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8) # Initialize the colored mask, mask.shape -> (128, 128), colored_mask.shape ->(128, 128, 3)

        for class_id in unique_classes: # Map each class value in the mask to a color from the colormap
            if class_id in colors:
                # Apply the color to the mask
                colored_mask[mask==class_id] = colors[class_id]

        return colored_mask


if __name__ == "__main__":
    _ = BratsLitModule(None, None, None, None)