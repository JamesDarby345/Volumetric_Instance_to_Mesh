import os
import nrrd
import numpy as np
import argparse
from midline_helper_simplified import filter_and_reassign_labels, create_slicer_nrrd_header, morphological_tunnel_filling
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

"""
Filters and reassign labels to remove small objects and
reassigns labels sequentially.
Useful preprocessing step for other methods in the repo to save computation time
and improve results due to removal of small artefacts from manual cube labelling.
"""

def process_file(file_path, morphological_radius=None, apply_filter=True):
    try:
        # Load the NRRD file
        data, header = nrrd.read(file_path)
       
        if apply_filter:
            processed_data = filter_and_reassign_labels(data, cc_min_size=200)
        else:
            processed_data = data

        # print(np.unique(processed_data))
        morphed_data = processed_data
        if morphological_radius:
            for label_val in np.unique(morphed_data):
                if label_val == 0:
                    continue
                # print(label_val)
                temp_morph_data = morphed_data == label_val
                temp_label_data = morphological_tunnel_filling(temp_morph_data, label_val=label_val, radius=morphological_radius)
                morphed_data = np.where((morphed_data == 0) & (temp_label_data > 0), label_val, morphed_data)

        # Extract z, y, x from the file path
        folder_name = os.path.basename(os.path.dirname(file_path))
        z, y, x = map(int, folder_name.split('_')[:3])

        header = create_slicer_nrrd_header(morphed_data, None, z, y, x)
        
        # Save the processed data back to the same file, with a new name if morphological radius is applied
        if morphological_radius:
            file_path = file_path.replace('_mask.nrrd', '_morph_mask.nrrd')
        nrrd.write(file_path, morphed_data, header)
        
        return None  # Return None for successful processing
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def process_mask_files(directory_path, morphological_radius, test_cubes=None, apply_filter=True):
    mask_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('_mask.nrrd') and not file.endswith('_relabeled_mask.nrrd') and not file.endswith('_morph_mask.nrrd'):
                file_path = os.path.join(root, file)
                if test_cubes:
                    # Check if the file path contains any of the test cube coordinates
                    if any(cube in file_path for cube in test_cubes):
                        mask_files.append(file_path)
                else:
                    mask_files.append(file_path)
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, mask_files, [morphological_radius] * len(mask_files), [apply_filter] * len(mask_files)), 
                            total=len(mask_files), desc="Processing files"))
    
    # Print only the error messages
    errors = [result for result in results if result is not None]
    for error in errors:
        print(error)

if __name__ == "__main__":
    # Default path
    default_path = "/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um"
    parser = argparse.ArgumentParser(description="Process mask files in a directory.")
    parser.add_argument("--path", type=str, help="Path to the directory containing mask files")
    parser.add_argument("--morph", type=int, help="Morphological tunnel filling radius")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only specific test cubes)")
    parser.add_argument("--no-filter", action="store_true", help="Skip filter and reassign labels")  # Added argument
    args = parser.parse_args()

    test_cubes = None
    if args.test:
        test_cubes = ['01744_02256_03792', '01744_02256_04048', '01744_02256_04304', '01744_02000_04048', '01744_02000_04304']

    # Use the command line argument if provided, otherwise use the default path
    directory_path = args.path if args.path else default_path
    morphological_radius = args.morph if args.morph else None
    apply_filter = not args.no_filter  # Determine whether to apply filtering
    process_mask_files(directory_path, morphological_radius, test_cubes, apply_filter)
