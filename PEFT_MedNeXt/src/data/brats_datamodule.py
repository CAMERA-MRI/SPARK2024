from typing import Any, Dict, Optional, Tuple
import torch
# from torch import device
# from torch._C import device
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from .components.preparing_data import get_volumes_path
# from .components.preparing_data import BratsDataset
# from .components.data_transforms import permute_and_add_axis_to_mask, spatialpad

import torchio as tio


class BratsDataModule(LightningDataModule):
    """LightningDataModule Implements 7 key methods"""
    def __init__(self,
                 data_dir: str = "data/", # data_directory. Defaults to "data/"
                 batch_size: int=1, # Batch size for the dataloaders, defaults to 1
                 num_workers: int=0, # Number of worker processes for data loading, defaults to 0
                 pin_memory: bool = False, # If True, the data loader will copy Tensors into CUDA pinned memory before returning them, defaults to False
                 ) -> None:
        
        super().__init__()

        # This allows to access init params with "self.hparams" attribute. also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir # Directory for data
        self.batch_size = batch_size 

        # data transformations
        self.train_transform = tio.Compose([ # input of shape (B, C, 240, 240, 155) goes through it
            tio.ToCanonical(), # Ensures same orientation of RAS
            tio.Resize(target_shape=(64, 64, 64)), # this transform should not be used as it will deform the physical object by scaling anistropically along the different dimensions. The solution to change an image size is typically applying Resample and CropOrPad.
            # tio.Resample(1),
            # tio.CropOrPad(target_shape=(256, 256, 256), mask_name='mask'),
            tio.ZNormalization() # Ensures resulting distribution of zero mean and unit SD.
        ]) # Output of shape (BCHWD) -> (B, 4, 256, 256, 256)

        self.val_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resize(target_shape=(64, 64, 64)),
            # tio.Resample(1),
            # tio.CropOrPad(target_shape=(256, 256, 256), mask_name='mask'),
            tio.ZNormalization(),
        ])

        self.test_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resize(target_shape=(64, 64, 64)),
            # tio.Resample(1),
            # tio.CropOrPad(target_shape=(256, 256, 256), mask_name='mask'),
            tio.ZNormalization()
        ])

        self.train_subjects = []
        self.val_subjects = []
        self.test_subjects = []

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None: # can't make state assignments, only used for task like downloading dataset
        pass

    def setup(self, stage=None) -> None: # 'stage' arg is used to separate logic for 'fit' and 'test', runs across all GPUs and is safe to make state assignments
        # Get paths to data volumes and segmentations. For now, test_image_paths and test_segmentation_paths also contains the validation dataset used for generating segmentations for validation submission
        train_image_paths, train_segmentation_paths, val_image_paths, val_segmentation_paths, test_image_paths, test_segmentation_paths = get_volumes_path(self.data_dir)
        
        # Instantiate Subject class and an empty list to accumulate all the subjects for training set
        # Each subject's image of shape (4, 240, 240, 155) and mask of shape (1, 240, 240, 155)
        for (image_path, segmentation_path) in zip(train_image_paths, train_segmentation_paths):
            train_subject = tio.Subject(
                image = tio.ScalarImage(image_path),
                mask = tio.LabelMap(segmentation_path),
                path = image_path, # path contains case_id which is required later in generating the prediction files. 
            )
            self.train_subjects.append(train_subject)

        # Instantiate Subject class and an empty list to accumulate all the subjects for validation set
        for (image_path, segmentation_path) in zip(val_image_paths, val_segmentation_paths):
            val_subject = tio.Subject(
                image = tio.ScalarImage(image_path),
                mask = tio.LabelMap(segmentation_path),
                path = image_path,
            )
            self.val_subjects.append(val_subject)

        # Instantiate Subject class and an empty list to accumulate all the subjects for test set
        for (image_path, segmentation_path) in zip(test_image_paths, test_segmentation_paths):
            test_subject = tio.Subject(
                image = tio.ScalarImage(image_path),
                mask = tio.LabelMap(segmentation_path),
                path = image_path

            )
            self.test_subjects.append(test_subject)

        # Assign training and validation datasets for the 'fit' stage
        if stage == "fit" or stage is None:
            self.data_train = tio.SubjectsDataset(self.train_subjects, transform=self.train_transform)
            self.data_val = tio.SubjectsDataset(self.val_subjects, transform=self.val_transform)
        
        # Assign test dataset for the 'test' stage
        if stage == 'test' or stage is None:
            self.data_test = tio.SubjectsDataset(self.test_subjects, transform=self.test_transform)

    def train_dataloader(self): # Method to create and return the training dataloader
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )
    
    # def predict_dataloader(self) -> Any:
    #     return DataLoader(datasset=self.data_predict,
    #                       batch_size=self.batch_size,
    #                       num_workers=self.hparams.num_workers,
    #                       pin_memory=self.hparams.pin_memory,
    #                       shuffle=False,
    #                       drop_last=True
    #                              )
    
    def teardown(self, stage): # Method to clean up after each stage
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        batch['image'][tio.DATA] = batch['image'][tio.DATA].to(device)
        batch['mask'][tio.DATA] = batch['mask'][tio.DATA].to(device)
        # batch['path'] = batch['path'].to(device)
        return batch
        # return super().transfer_batch_to_device(batch, device, dataloader_idx)

if __name__ == "__main__":
    _ = BratsDataModule()