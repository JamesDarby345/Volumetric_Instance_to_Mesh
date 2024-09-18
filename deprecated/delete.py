import os
import argparse
import glob
import shutil

def delete_files_and_folders(directory):
    # Delete specific .nrrd files
    pattern = os.path.join(directory, '**', '*_relabeled_fb_avg_mask_thinned.nrrd')
    files = glob.glob(pattern, recursive=True)
    
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except OSError as e:
            print(f"Error deleting file {file}: {e}")
    
    print(f"Total files deleted: {len(files)}")

    # Delete /obj folders and their contents
    obj_folders = glob.glob(os.path.join(directory, '**/obj'), recursive=True)
    
    for folder in obj_folders:
        try:
            shutil.rmtree(folder)
            print(f"Deleted folder: {folder}")
        except OSError as e:
            print(f"Error deleting folder {folder}: {e}")
    
    print(f"Total obj folders deleted: {len(obj_folders)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete *_relabeled_fb_avg_mask_thinned.nrrd files and /obj folders recursively")
    parser.add_argument("--dir", help="Input directory to search for files and folders")
    args = parser.parse_args()

    delete_files_and_folders(args.dir)