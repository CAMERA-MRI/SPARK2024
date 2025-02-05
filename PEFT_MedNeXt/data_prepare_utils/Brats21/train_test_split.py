import os
import shutil
import random
import argparse

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_folders(src, dst, folders):
    for folder in folders:
        src_folder = os.path.join(src, folder)
        dst_folder = os.path.join(dst, folder)
        shutil.copytree(src_folder, dst_folder)

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Split dataset into training and testing sets.")
    
    # Add arguments for source directory, training directory, and validation directory
    parser.add_argument('--source_dir', type=str, required=True, help='Source directory containing the dataset i.e. base directory having n folders for n patients')
    parser.add_argument('--train_dir', type=str, default=os.path.join(os.getcwd(), 'train_subset'), required=False, help='Directory to store the training subset')
    parser.add_argument('--test_dir', type=str, default=os.path.join(os.getcwd(), 'test_subset'), required=False, help='Directory to store the validation subset')
    parser.add_argument('--train_ratio', type=float, default=0.8, required=True, help='Split Size between 0 and 1. For example, 0.8 splits 80 percent data into training set and rest for validation set')
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    source_dir = args.source_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    split_size = args.train_ratio

    # # Define source and destination directories
    # source_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\BraTS2023_SSA_Training'
    # train_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\train_subset'
    # test_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\test_subset'

    # Create destination directories if they don't exist
    create_directory(train_dir)
    create_directory(test_dir)

    # Get the list of all folders in the source directory
    all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # Shuffle the list of folders
    random.shuffle(all_folders)

    # Determine the number of folders for training and validation
    train_size = int(split_size * len(all_folders))
    train_folders = all_folders[:train_size]
    validation_folders = all_folders[train_size:]

    # Copy the folders to the respective destination directories
    copy_folders(source_dir, train_dir, train_folders)
    copy_folders(source_dir, test_dir, validation_folders)

    print(f"Folders successfully splitted. Stored training subset in {train_dir} and validation subset in {test_dir}")

if __name__ == '__main__':
    main()

