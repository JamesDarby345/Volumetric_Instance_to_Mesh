import os
import nrrd
import numpy as np
import argparse
from midline_helper_simplified import filter_and_reassign_labels
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

"""
Filters and reassign labels to remove small objects and
reassigns labels sequentially.
Useful preprocessing step for other methods in the repo to save computation time
and improve results due to removal of small artefacts from manual cube labelling.
"""

def process_file(file_path, cc_min_size=200):
    try:
        # Load the NRRD file
        data, header = nrrd.read(file_path)
        
        # Apply filter_and_reassign_labels
        processed_data = filter_and_reassign_labels(data, cc_min_size=cc_min_size)
        
        # Save the processed data back to the same file
        nrrd.write(file_path, processed_data, header)
        
        return None  # Return None for successful processing
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def process_mask_files(directory):
    mask_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_mask.nrrd'):
                mask_files.append(os.path.join(root, file))
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, mask_files), total=len(mask_files), desc="Processing files"))
    
    # Print only the error messages
    errors = [result for result in results if result is not None]
    for error in errors:
        print(error)

if __name__ == "__main__":
    # Default path
    default_path = "/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um"
    parser = argparse.ArgumentParser(description="Process mask files in a directory.")
    parser.add_argument("--path", type=str, help="Path to the directory containing mask files")
    args = parser.parse_args()

    # Use the command line argument if provided, otherwise use the default path
    directory_path = args.path if args.path else default_path
    
    process_mask_files(directory_path)
