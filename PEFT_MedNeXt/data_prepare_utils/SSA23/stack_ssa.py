import os
import shutil
import nibabel as nib
import numpy as np
import argparse


def create_directory(dir_path):
    """Creates a directory if it doesn't exist."""
    if os.path.exists(dir_path):
        raise FileExistsError(f"Directory '{dir_path}' already exists.")
    else:
        os.makedirs(dir_path)
        print(f"Directory '{dir_path} created.")


def process_patient_folders(source_dir, destination_imagesTr, destination_labelsTr):
    # Create destination directories if they don't exist\
    create_directory(destination_imagesTr)
    create_directory(destination_labelsTr)

    i = 0
    # Traverse through each patient folder
    for i, patient_folder in enumerate(os.listdir(source_dir)):
        patient_folder_path = os.path.join(source_dir, patient_folder)
        
        # Skip if it's not a directory
        if not os.path.isdir(patient_folder_path):
            continue

        # Initialize a list to hold the images for stacking
        # modalities = ['flair', 't1', 't1ce', 't2']
        modalities = ['t1c', 't1n', 't2f', 't2w']
        images = []
        affine = None

        # Process each modality
        for modality in modalities:
            # modality_file = f"{patient_folder}_{modality}.nii.gz"
            modality_file = f"{patient_folder}-{modality}.nii.gz"
            modality_path = os.path.join(patient_folder_path, modality_file)
            
            if os.path.exists(modality_path):
                # Load the image and append to the list
                nii_image = nib.load(modality_path)
                image_data = nii_image.get_fdata().astype(np.float32)
                images.append(image_data)

                if affine is None:
                    # Get the affine matrix and append to the list
                    affine = nii_image.affine
            else:
                print(f"File {modality_path} not found, skipping modality {modality}.")
                continue    

        # Stack the images along a new dimension (4th dimension)
        if len(images) == len(modalities):
            stacked_image = np.stack(images, axis=-1)
            stacked_image_nii = nib.Nifti1Image(stacked_image, affine=affine)
            
            # Define the new filename for the stacked image
            stacked_image_filename = f"{patient_folder}_stacked.nii.gz"
            stacked_image_path = os.path.join(destination_imagesTr, stacked_image_filename)
            
            # Save the stacked image
            nib.save(stacked_image_nii, stacked_image_path)

        # Process the segmentation file
        # segmentation_file = f"{patient_folder}_seg.nii.gz"
        segmentation_file = f"{patient_folder}-seg.nii.gz"
        segmentation_path = os.path.join(patient_folder_path, segmentation_file)
        
        if os.path.exists(segmentation_path):
            new_segmentation_path = os.path.join(destination_labelsTr, f"{patient_folder}.nii.gz")
            shutil.copy(segmentation_path, new_segmentation_path)
        else:
            print(f"Segmentation file {segmentation_path} not found, skipping.")

        if i % 9 == 0:
            print(f'Processed {i+1} files')

        # # Remove the original patient folder
        # shutil.rmtree(patient_folder_path)
        i += 1
    print(f'Total of {i+1} Files processed; stacked images, and moved them successfully, along with the labels.')


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Stacks the four modalities into one.")
    
    # Add arguments for source directory, destination directories for image and labels
    parser.add_argument('--source_dir', type=str, required=True, help='Source directory containing the dataset i.e. base directory having n folders for n patients')
    parser.add_argument('--flag', type=str, required=True, choices=['train', 'val', 'test'], help="Specify either of these: 'train', 'validation', 'test'. 'train' represents you are doing stacking of the images on training set. For instance, if specified Flag as 'train', it will create TrainVolumes and TrainSegmentations folders and it will be used for storing stacked train images and segmentations. Similary, creates ValVolumes and ValSegmentations for validation set. Also, TestVolumes, TestSegmentations for testing set.")

    # Parse the command line arguments
    args = parser.parse_args()

    # Determine the destination directories based on the flag
    if args.flag == 'train':
        destination_images_dir = os.path.join(os.getcwd(), 'stacked', 'TrainVolumes')
        destination_labels_dir = os.path.join(os.getcwd(), 'stacked', 'TrainSegmentations')
    elif args.flag == 'val':
        destination_images_dir = os.path.join(os.getcwd(), 'stacked', 'ValVolumes')
        destination_labels_dir = os.path.join(os.getcwd(), 'stacked', 'ValSegmentations')
    elif args.flag == 'test':
        destination_images_dir = os.path.join(os.getcwd(), 'stacked', 'TestVolumes')
        destination_labels_dir = os.path.join(os.getcwd(), 'stacked', 'TestSegmentations') 
    else:
        raise ValueError(f"Invalid choice for --flag: '{args.flag}'. Must be one of 'train', 'val', 'test'.")
    
    # Call the function to process patient folders
    process_patient_folders(args.source_dir, destination_images_dir, destination_labels_dir)


if __name__ == '__main__':
    main()