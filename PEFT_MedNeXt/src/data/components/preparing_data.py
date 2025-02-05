import os
from glob import glob as glob
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset


def get_volumes_path(base_data_dir):
    train_image_paths = sorted(glob(os.path.join(base_data_dir, 'TrainVolumes', '*.nii.gz')))
    train_segmentation_paths = sorted(glob(os.path.join(base_data_dir, 'TrainSegmentations', '*.nii.gz')))
    
    val_image_paths = sorted(glob(os.path.join(base_data_dir, 'ValVolumes', '*.nii.gz')))
    val_segmentation_paths = sorted(glob(os.path.join(base_data_dir, 'ValSegmentations', '*.nii.gz')))

    test_image_paths = sorted(glob(os.path.join(base_data_dir, 'TestVolumes', '*.nii.gz')))
    test_segmentation_paths = sorted(glob(os.path.join(base_data_dir, 'TestSegmentations', '*.nii.gz')))

    def assert_image_and_segmentation_name_match(image_paths, segmentation_paths):
        for image_path, segmentation_path in zip(image_paths, segmentation_paths):
            assert (os.path.basename(image_path).split("_")[0] == os.path.basename(segmentation_path).split(".")[0], f'Image and Segmentation name mismatch: {image_path} != {segmentation_path}') or ('_'.join(os.path.basename(image_path).split("_")[0:2]) == os.path.basename(segmentation_path).split(".")[0], f'Image and Segmentation name mismatch: {image_path} != {segmentation_path}')

    assert_image_and_segmentation_name_match(train_image_paths, train_segmentation_paths)
    assert_image_and_segmentation_name_match(val_image_paths, val_segmentation_paths)
    assert_image_and_segmentation_name_match(test_image_paths, test_segmentation_paths)

    return train_image_paths, train_segmentation_paths, val_image_paths, val_segmentation_paths, test_image_paths, test_segmentation_paths


# class BratsDataset(Dataset):
#     def __init__(self, images_path_list, masks_path_list, transform=None):
#         """
#         Args:
#             images_path_list (list of strings): List of paths to input images.
#             masks_path_list (list of strings): List of paths to masks.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.images_path_list = images_path_list
#         self.masks_path_list = masks_path_list
#         self.transform = transform
#         self.length = len(images_path_list)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # Load image
#         image_path = self.images_path_list[idx]
#         image = nib.load(image_path).get_fdata()
#         image = np.float32(image) # shape of image [240, 240, 155, 4]

#         # Load mask
#         mask_path = self.masks_path_list[idx]
#         mask = nib.load(mask_path).get_fdata()
#         mask = np.float32(mask) # shape of mask [240, 240, 155]

#         if self.transform:
#             transformed_sample = self.transform({'image': image, 'mask': mask})
        
#         return transformed_sample
