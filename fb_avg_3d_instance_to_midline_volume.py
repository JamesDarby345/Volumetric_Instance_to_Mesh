import os
import numpy as np
import nrrd
import time
import concurrent.futures
import argparse
from tqdm import tqdm
import threading

from midline_helper_simplified import *


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

def process_single_label(label_data, label_value, front=True, back=True, mask_out=True):
    mask = (label_data == label_value)
    result = front_back_avg_of_structure(mask, front, back)
    if mask_out:
        result[mask != 1] = 0
    return result

def process_single_label_wrapper(args):
    data, label_val, pad_amount, rotation_info, front, back, mask_out = args
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

def process_structures(nrrd_path, output_path, pad_amount=10, label_values=None, minObjSize=200, filter_labels=False, front=True, back=True, mask_out=True):
    try:        
        original_data, header = nrrd.read(nrrd_path)
        
        if filter_labels:
            original_data = filter_and_reassign_labels(original_data, minObjSize)
        
        midline_labels = np.zeros_like(original_data, dtype=np.uint8)

        if label_values:
            unique_labels = label_values
        else:
            unique_labels = np.unique(original_data)
            unique_labels = unique_labels[unique_labels != 0]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
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
                
                args = (data, label_val, pad_amount, rotation_info, front, back, mask_out)
                futures.append(executor.submit(process_single_label_wrapper, args))

            for future in concurrent.futures.as_completed(futures):
                thinned_data = future.result()
                midline_labels = np.maximum(midline_labels, thinned_data.astype(np.uint8))
        
        # Start a background thread for writing NRRD
        write_thread = threading.Thread(target=write_nrrd_background, args=(output_path, midline_labels.astype(np.uint8), header))
        write_thread.start()
        
        return midline_labels, write_thread
    except Exception as e:
        print(f"Error processing {nrrd_path}: {str(e)}")
        return None, None

def process_directory(input_directory, pad_amount=0, label_values=None, test_mode=False, replace=False, show_time=False, filter_labels=False, front=True, back=True, mask_out=True, use_relabeled=False):
    files_to_process = []
    write_threads = []
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
            
            files_to_process.append((input_path, output_path))
            
            if test_mode and len(files_to_process) >= 3:
                break

    for input_path, output_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        overall_start_time = time.time()
        
        result, write_thread = process_structures(input_path, output_path, pad_amount, label_values, filter_labels=filter_labels, front=front, back=back, mask_out=mask_out)
        
        if result is None and show_time:
            print(f"Error processing file: {os.path.basename(input_path)}")
        elif show_time:
            print(f"Time taken for file {os.path.basename(input_path)}: {time.time() - overall_start_time:.2f} seconds")
        
        if write_thread:
            write_threads.append(write_thread)

    # Wait for all write operations to complete
    for thread in write_threads:
        thread.join()

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
    args = parser.parse_args()

    # If neither front nor back is specified, use both and average
    if not args.front and not args.back:
        args.front = True
        args.back = True

    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_dir if args.input_dir else default_input_directory
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    process_directory(input_directory, pad_amount=0, label_values=label_values, test_mode=args.test, 
                      replace=args.replace, show_time=args.time, filter_labels=args.filter_labels, 
                      front=args.front, back=args.back, mask_out=True, use_relabeled=args.relabeled)
    
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")