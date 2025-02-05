import numpy as np

##################################
# Data transforms
###################################

class permute_and_add_axis_to_mask(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = image.transpose((3, 2, 0, 1)) # Before: [HWDC], After: [CDHW]
        mask = mask.transpose((2, 0, 1)) # Before [HWD], After: [DHW]

        mask= mask[np.newaxis, ...] #[1,D,H,W]
        return {'image':image,
                'mask':mask}
    
class spatialpad(object): # First dimension should be left untouched of [C, D, H, W]
    def __init__(self, image_target_size=[4, 256, 256, 256], mask_target_size=[1, 256, 256, 256]):
        self.image_target_size = image_target_size
        self.mask_target_size = mask_target_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'] # image: [4, 155, 240, 240], mask[1, 155, 240, 240]

        padded_image = self.pad_input(image, self.image_target_size)

        padded_mask = self.pad_input(mask, self.mask_target_size)

        return {'image': padded_image, # [4, 256, 256, 256]
                'mask': padded_mask} #[1, 256, 256, 256]
    

    def pad_input(self, input_array, target_size):
        # Ensure the input array is a numpy array
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        # Calculate padding sizes for each dimension
        pad_width = []
        for i in range(len(input_array.shape)):
            total_padding = target_size[i] - input_array.shape[i]
            if total_padding < 0:
                raise ValueError(f"Target shape must be larger than the input shape. Dimension {i} is too small.")
            pad_before = total_padding // 2
            pad_after = total_padding - pad_before
            pad_width.append((pad_before, pad_after))

        # Pad the image
        padded_image = np.pad(input_array, pad_width, mode='constant', constant_values=0)

        return padded_image   

