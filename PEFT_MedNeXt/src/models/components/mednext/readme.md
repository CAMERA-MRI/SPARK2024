### What's here
##### blocks.py
It include all convolution layer blocks for mednext architecture

##### MedNeXt.py
It is the main architecture block which implement Mexnext architecture

###### create_mednext.py
It include functions for small, medium, large and custom mednexr arcxhitecture
__Small__ : def create_mednextv1_small(num_input_channels, num_classes, kernel_size=3, ds=False)
__base__ : def create_mednextv1_base(num_input_channels, num_classes, kernel_size=3, ds=False)
__medium__: def create_mednextv1_medium(num_input_channels, num_classes, kernel_size=3, ds=False)
__large__: def create_mednextv1_large(num_input_channels, num_classes, kernel_size=3, ds=False)
__custom__: def create_mednextv1_custom(num_input_channels, num_classes, kernel_size=3, ds=False)