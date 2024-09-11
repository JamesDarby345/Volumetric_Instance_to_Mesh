import os
import numpy as np
import nrrd
import time
import concurrent.futures
import graph_tool.all as gt
import argparse
from tqdm import tqdm
import threading

from midline_helper_simplified import *
from deprecated.distance_map_utils import *

"""
Faster than graph construction method, slower than front back average method,
doesnt fill holes (morphological tunnels).
Constructs distance map for each label
Uses single value assumption along maximum PCA direction to take highest dist map
value at each point to create the midline volume.
Follows midline more closely on aggresive curves.
Morphological tunnels are filled during the mesh creation process.
Leads to closer midlines, but rougher labels, especially around holes.
Should have a low chance of causing obj collisions as the midline cannot
leave the instance label.
"""

def process_single_label(label_data, label_value, output_path):
    mask = (label_data == label_value)
    distance_map = create_single_label_distance_map(mask, label_value)
    result = dist_map_max(distance_map)
    return result

def process_single_label_wrapper(args):
    data, label_val, output_path, pad_amount, rotation_info = args
    thinned_data = process_single_label(data, label_val, output_path)
    
    if pad_amount:
        pad_amount += 1
        thinned_data = thinned_data[pad_amount:-pad_amount, pad_amount:-pad_amount, pad_amount:-pad_amount]
        thinned_data = np.pad(thinned_data, 1, mode='constant', constant_values=0)
    
    thinned_data = unapply_rotation(thinned_data, rotation_info)
    thinned_data[thinned_data != 0] = label_val
    return thinned_data

def write_nrrd_background(output_path, data, header):
    nrrd.write(output_path, data, header)

def process_structures(nrrd_path, output_path, pad_amount=10, label_values=None, minObjSize=200, filter_labels=False):
    try:
        original_data, header = nrrd.read(nrrd_path)
        if filter_labels:
            print("Filtering and reassigning labels")
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
                    # print(f"Label {label_val} has no voxels, skipping")
                    continue
                # print(f"Label {label_val} has {np.sum(mask)} voxels")
                
                if pad_amount:
                    data = np.pad(data, pad_amount, mode='constant', constant_values=0)
                    data = connect_to_edge_3d(data, label_val, pad_amount+1, use_z=True, create_outline=False)
                
                args = (data, label_val, output_path, pad_amount, rotation_info)
                futures.append(executor.submit(process_single_label_wrapper, args))

            
            for future in concurrent.futures.as_completed(futures):
                thinned_data = future.result()
                midline_labels += thinned_data.astype(np.uint8)
        
        # Start a background thread for writing the NRRD file
        write_thread = threading.Thread(target=write_nrrd_background, args=(output_path, midline_labels.astype(np.uint8), header))
        write_thread.start()

        return midline_labels, write_thread
    except Exception as e:
        print(f"Error processing {nrrd_path}: {str(e)}")
        return None, None

def process_directory(input_directory, pad_amount=0, label_values=None, test_mode=False, replace=False, show_time=False, filter_labels=False):
    files_to_process = []
    for root, _, files in os.walk(input_directory):
        mask_file = next((f for f in files if f.endswith('_mask.nrrd')), None)
        if mask_file:
            input_path = os.path.join(root, mask_file)
            output_path = input_path.replace('_mask.nrrd', '_dist_map_mask_thinned.nrrd')
            
            if os.path.exists(output_path) and not replace:
                continue
            
            files_to_process.append((input_path, output_path))
            
            if test_mode and len(files_to_process) >= 1:
                break

    write_threads = []
    for input_path, output_path in tqdm(files_to_process, desc="Processing files", unit="file"):
        overall_start_time = time.time()
        
        result, write_thread = process_structures(input_path, output_path, pad_amount, label_values, filter_labels=filter_labels)
        
        if result is None and show_time:
            print(f"Error processing file: {os.path.basename(input_path)}")
        elif show_time:
            print(f"Time taken for file {os.path.basename(input_path)}: {time.time() - overall_start_time:.2f} seconds")
        
        if write_thread:
            write_threads.append(write_thread)

    # Wait for all write threads to complete
    for thread in write_threads:
        thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D instance volumes to midline volumes.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only first file)')
    parser.add_argument('--replace', action='store_true', help='Replace existing _dist_map_mask_thinned.nrrd files')
    parser.add_argument('--time', action='store_true', help='Show execution time for each file')
    parser.add_argument('--filter-labels', action='store_true', help='Filter and reassign labels')
    args = parser.parse_args()

    current_directory = os.getcwd()
    input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    process_directory(input_directory, pad_amount=0, label_values=label_values, test_mode=args.test, replace=args.replace, show_time=args.time, filter_labels=args.filter_labels)
    
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")