import os
import argparse
import glob

def delete_files(directory):
    pattern = os.path.join(directory, '**', '*_relabeled_fb_avg_mask_thinned.nrrd')
    files = glob.glob(pattern, recursive=True)
    
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"Total files deleted: {len(files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete *_relabeled_fb_avg_mask_thinned.nrrd files recursively")
    parser.add_argument("--dir", help="Input directory to search for files")
    args = parser.parse_args()

    delete_files(args.dir)