import os
import re
import numpy as np
import nrrd
import time
import concurrent.futures
import argparse
from tqdm import tqdm
from midline_helper_simplified import *
from scipy.ndimage import binary_erosion

import fastmorph

"""
Fastest midline calculation method.
Front and back of the instance volume are averaged to create a midline volume.
PCA components are used to determine the orientation of the structure.
Doesnt fill holes.
Disconnected fibres in the same label are an issue for this method as the
front back average ends up out in space off of the sheet.
Follows midline more closely on aggresive curves.
Morphological tunnels are filled during the mesh creation process.
Leads to closer midlines, but rougher labels, especially around holes.
The front back average could leave the instance label, causing obj collisions 
with low probability.
"""

@numba.jit(nopython=True, parallel=True)
def numba_dilation_3d_labels(data, iterations):
    result = data.copy()
    rows, cols, depths = data.shape
    
    for _ in range(iterations):
        temp = result.copy()
        for i in numba.prange(rows):
            for j in range(cols):
                for k in range(depths):
                    if result[i, j, k] == 0:  # Only dilate into empty space
                        # Check 6-connected neighbors
                        if i > 0 and temp[i-1, j, k] != 0:
                            result[i, j, k] = temp[i-1, j, k]
                        elif i < rows-1 and temp[i+1, j, k] != 0:
                            result[i, j, k] = temp[i+1, j, k]
                        elif j > 0 and temp[i, j-1, k] != 0:
                            result[i, j, k] = temp[i, j-1, k]
                        elif j < cols-1 and temp[i, j+1, k] != 0:
                            result[i, j, k] = temp[i, j+1, k]
                        elif k > 0 and temp[i, j, k-1] != 0:
                            result[i, j, k] = temp[i, j, k-1]
                        elif k < depths-1 and temp[i, j, k+1] != 0:
                            result[i, j, k] = temp[i, j, k+1]
                        
    return result

def morphological_tunnel_filling(arr, label_val, radius=10):
    arr = (arr > 0).astype(bool)
    arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=0)

    if radius > 20: 
        arr = fastmorph.spherical_dilate(arr, radius=radius, parallel=1, in_place=True)
        arr = fastmorph.spherical_erode(arr, radius=radius, parallel=1, in_place=True)
    else:
        arr = numba_dilation_3d_labels(arr, iterations=radius)
        arr = binary_erosion(arr, iterations=radius)

    arr = arr[radius:-radius, radius:-radius, radius:-radius]
    arr = np.where(arr > 0, label_val, 0)

    return arr

def front_back_avg_of_structure(arr, front=True, back=True):
    # Get the dimensions of the input array
    x_dim, y_dim, z_dim = arr.shape
    
    # Create an output array of the same dimensions, filled with zeros
    output = np.zeros_like(arr)
    
    
    # Iterate through each x,y line
    for x in range(x_dim):
        for y in range(y_dim):
            # Get the current line
            line = arr[x, y, :]
            
            # Find non-zero indices
            non_zero_indices = np.nonzero(line)[0]
            
            # If there are non-zero elements in the line
            if len(non_zero_indices) > 0:
                first_non_zero = non_zero_indices[0]
                last_non_zero = non_zero_indices[-1]
                
                if front and back:
                    # Calculate the average index
                    index = int((first_non_zero + last_non_zero) / 2)
                elif front:
                    index = first_non_zero
                elif back:
                    index = last_non_zero
                else:
                    continue
                
                # Set the chosen position to 1 in the output array
                output[x, y, index] = 1
    
    return output

def process_single_label(label_data, label_value, front=True, back=True, mask_out=True):
    mask = (label_data == label_value)
    
    # Calculate front-back average for each axis rotation
    # results = []
    # for axis in range(3):
    #     rotated_mask = np.rot90(mask, k=1, axes=(axis, (axis+1)%3))
    #     result = front_back_avg_of_structure(rotated_mask, front, back)
    #     result = np.rot90(result, k=-1, axes=(axis, (axis+1)%3))
    #     results.append(result)

    best_result = front_back_avg_of_structure(mask, front, back)
    
    # Count voxels outside the original label for each result
    # outside_voxels = [np.sum((r > 0) & (mask == 0)) for r in results]

    # #Count the voxels in the original mask for each result
    # inside_voxels = [np.sum((r > 0) & (mask == 1))for r in results]
    
    # Choose the result with the fewest outside voxels
    # best_result = results[np.argmin(outside_voxels)]
    # best_result = results[np.argmax(inside_voxels)]
    # Sum up all the result arrays
    # combined_result = np.sum(results, axis=0)
    # best_result = combined_result
    
    # Remove values outside the original structure if mask_out is True
    if mask_out:
        best_result[mask != 1] = 0
    
    return best_result

def process_single_label_wrapper(args):
    data, label_val, pad_amount, rotation_info, front, back, mask_out, morph=args
    if morph > 0:
        data = morphological_tunnel_filling(data, label_val, radius=morph)
    thinned_data = process_single_label(data, label_val, front, back, mask_out).astype(np.uint8)
    
    if pad_amount:
        pad_amount += 1
        thinned_data = thinned_data[pad_amount:-pad_amount, pad_amount:-pad_amount, pad_amount:-pad_amount]
        thinned_data = np.pad(thinned_data, 1, mode='constant', constant_values=0)
    
    thinned_data = unapply_rotation(thinned_data, rotation_info)
    thinned_data[thinned_data != 0] = label_val
    return thinned_data

def write_nrrd_background(output_path, data, header):
    nrrd.write(output_path, data, header)

def process_structures(nrrd_path, output_path, pad_amount=10, label_values=None, minObjSize=200, filter_labels=False, front=True, back=True, mask_out=True, morph=0):
    # try:        
        original_data, header = nrrd.read(nrrd_path)
        
        if filter_labels:
            original_data = filter_and_reassign_labels(original_data, minObjSize)
        
        midline_labels = np.zeros_like(original_data, dtype=np.uint8)

        if label_values:
            unique_labels = label_values
        else:
            unique_labels = np.unique(original_data)
            unique_labels = unique_labels[unique_labels != 0]
        
        for label_val in unique_labels:
            data = original_data.copy()
            primary_label_direction = calculate_orientation(data, label_val)
            data, rotation_info = rotate_to_z_axis(data, primary_label_direction)
            mask = data == label_val
            data[mask != 1] = 0
            
            if np.sum(mask) == 0:
                continue
            
            if pad_amount:
                data = np.pad(data, pad_amount, mode='constant', constant_values=0)
                data = connect_to_edge_3d(data, label_val, pad_amount+1, use_z=True, create_outline=False)
            
            args = (data, label_val, pad_amount, rotation_info, front, back, mask_out, morph)
            thinned_data = process_single_label_wrapper(args)
            midline_labels = np.maximum(midline_labels, thinned_data.astype(np.uint8))
        
        # Write NRRD file
        nrrd.write(output_path, midline_labels.astype(np.uint8), header)
        
        return midline_labels
    # except Exception as e:
    #     print(f"Error processing {nrrd_path}: {str(e)}")
    #     return None

def process_single_file(args):
    input_path, output_path, pad_amount, label_values, filter_labels, front, back, mask_out, morph= args
    start_time = time.time()
    result = process_structures(input_path, output_path, pad_amount, label_values, filter_labels=filter_labels, front=front, back=back, mask_out=mask_out, morph=morph)
    processing_time = time.time() - start_time
    return input_path, result, processing_time

def process_directory(input_directory, pad_amount=0, label_values=None, test_mode=[], replace=False, show_time=False, filter_labels=False, front=True, back=True, mask_out=True, use_relabeled=False, morph=0):
    files_to_process = []
    for root, _, files in os.walk(input_directory):
        mask_file_suffix = '_relabeled_mask.nrrd' if use_relabeled else '_mask.nrrd'
        mask_file = next((f for f in files if f.endswith(mask_file_suffix)), None)
        if mask_file:
            input_path = os.path.join(root, mask_file)
            if front and not back:
                output_path = input_path.replace(mask_file_suffix, '_front_mask_thinned.nrrd')
            elif back and not front:
                output_path = input_path.replace(mask_file_suffix, '_back_mask_thinned.nrrd')
            else:
                output_path = input_path.replace(mask_file_suffix, '_fb_avg_mask_thinned.nrrd')
            
            if os.path.exists(output_path) and not replace:
                continue
            
            files_to_process.append((input_path, output_path, pad_amount, label_values, filter_labels, front, back, mask_out, morph))
            
            if len(test_mode) > 0:
                # Filter files_to_process based on test_mode
                files_to_process = [
                    args for args in files_to_process 
                    if any(re.search(pattern, args[0]) for pattern in test_mode)
                ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, args) for args in files_to_process]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Processing files", unit="file"):
            input_path, result, processing_time = future.result()
            if result is None and show_time:
                print(f"Error processing file: {os.path.basename(input_path)}")
            elif show_time:
                print(f"Time taken for file {os.path.basename(input_path)}: {processing_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D instance volumes to midline volumes.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only first 3 files)')
    parser.add_argument('--replace', action='store_true', help='Replace existing _mask_thinned.nrrd files')
    parser.add_argument('--time', action='store_true', help='Show execution time for each file')
    parser.add_argument('--filter-labels', action='store_true', help='Filter and reassign labels')
    parser.add_argument('--front', action='store_true', help='Use front of structure')
    parser.add_argument('--back', action='store_true', help='Use back of structure')
    parser.add_argument('--input-dir', type=str, help='Input directory path')
    parser.add_argument('--relabeled', action='store_true', help='Process _relabeled_mask.nrrd files instead of _mask.nrrd')
    parser.add_argument('--no-mask-out', action='store_true', help='Do not mask out values that arent part of the structure')
    parser.add_argument('--morph', type=int, default=0, help='Use morphological dilation and erosion to close holes with provided radius, recomend 10, takes ~2x longer')
    args = parser.parse_args()

    # If neither front nor back is specified, use both and average
    if not args.front and not args.back:
        args.front = True
        args.back = True

    if args.test:
        args.test = ['01744_02256_04048', '01744_02256_04304']
    else:
        args.test = []

    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_dir if args.input_dir else default_input_directory
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    process_directory(input_directory, pad_amount=0, label_values=label_values, test_mode=args.test, 
                      replace=args.replace, show_time=args.time, filter_labels=args.filter_labels, 
                      front=args.front, back=args.back, mask_out=not args.no_mask_out, use_relabeled=args.relabeled, morph=args.morph)
    
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")