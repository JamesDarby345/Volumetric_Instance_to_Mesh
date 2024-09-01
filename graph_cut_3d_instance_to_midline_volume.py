import os
import numpy as np
import nrrd
import time
import concurrent.futures
import graph_tool.all as gt
import argparse
from tqdm import tqdm

from midline_helper_simplified import *
from distance_map_utils import (
    create_label_distance_map_with_roi,
    prepare_distance_map
)

"""
Slow due to graph construction which adds valid seam/sheet prior.
Results in 'smoother' midlines, but less accurate to the original midline
volume. When the label changes directions aggresively, the midline can
end up curving away from the original midline volume.
Single value assumption along maximum PCA direction.
Can only create voxels at maximum 45 degree angle to previous voxels,
limiting ability to follow aggresive curves.
Fill holes (morphological tunnels) well by using the graph construction 
to provide structure across the gap.
Could cause obj collisions as midline labels can leave their instance label
to allow for hole crossing, and end up inside of other objects.
"""

def calculate_seam_iter(directed_graph, src, tgt, weights, test_size, x_pos, y_pos, z_pos):
    res = gt.boykov_kolmogorov_max_flow(directed_graph, src, tgt, weights)
    flow = sum(weights[e] - res[e] for e in tgt.in_edges())
    part = gt.min_st_cut(directed_graph, src, weights, res)
    boundary_vertices = find_boundary_vertices(np.array(directed_graph.get_edges()), part)
    shape = (test_size, test_size, test_size)
    boundary_array = boundary_vertices_to_array_masked(boundary_vertices, shape, 'x', x_pos, y_pos, z_pos)
    return boundary_array, flow

def process_single_label(label_data, label_value, output_path):
    mask = (label_data == label_value)
    save_mask = mask.astype(np.uint8)
    # nrrd.write(f"/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/test_mask.nrrd", save_mask, {})
    
    stime = time.time()
    roi_mask = generate_volume_roi(mask, erode_dilate_iters=30)
    # print(f"Time taken to process ROI: {time.time() - stime:.2f} seconds")

    # nrrd.write(f"/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/test_roi.nrrd", roi_mask, {})
    
    stime = time.time()
    distance_map = create_label_distance_map_with_roi(mask, roi_mask)
    # print(f"Time taken to calculate distance map: {time.time() - stime:.2f} seconds")
    
    distance_map = prepare_distance_map(distance_map, roi_mask, value_to_add=10)

    # nrrd.write(f"/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/test_distance_map.nrrd", distance_map, {})
    
    stime = time.time()
    directed_graph, src, tgt, weights, x_pos, y_pos, z_pos = create_masked_directed_energy_graph_from_dist_map(distance_map)
    # print(f"Time taken to create energy graph: {time.time() - stime:.2f} seconds")
    
    stime = time.time()
    seam_array, _ = calculate_seam_iter(directed_graph, src, tgt, weights, distance_map.shape[0], x_pos, y_pos, z_pos)
    # print(f"Time taken to calculate seam: {time.time() - stime:.2f} seconds")

    # nrrd.write(f"/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/test_seam.nrrd", seam_array, {})

    return seam_array

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

#TODO: fix this
def check_for_intersections(label_list):
    # Create a combined volume to check for intersections
    combined = np.zeros_like(label_list[0], dtype=np.uint8)
    for label_array in label_list:
        combined[label_array != 0] += 1
    
    # Find intersections
    intersections = combined > 1
    
    # If no intersections, return the original list
    if not np.any(intersections):
        return label_list
    
    # Process each label to remove intersections
    for i, label_array in enumerate(label_list):
        mask = label_array != 0
        overlap = mask & intersections
        
        if np.any(overlap):
            # Find the non-overlapping part
            non_overlap = mask & ~intersections
            
            # Dilate the non-overlapping part to fill the gap
            from scipy import ndimage
            dilated = ndimage.binary_dilation(non_overlap)
            
            # Update the label array
            label_list[i] = np.where(dilated, label_array, 0)
    
    return label_list

def process_structures(nrrd_path, output_path, pad_amount=10, label_values=None, minObjSize=200, show_time=False, filter_labels=False):
    overall_start_time = time.time()
    
    try:
        original_data, header = nrrd.read(nrrd_path)
        if header['dimension'] != 3:
            print(f"Error: NRRD file {nrrd_path} does not have 3 dimensions. Skipping.")
            return None
    except Exception as e:
        print(f"Error reading NRRD file: {e}")
        return None

    try:
        if filter_labels:
            original_data = filter_and_reassign_labels(original_data, minObjSize)
        midline_labels = np.zeros_like(original_data, dtype=np.uint8)
        label_list = []

        if label_values:
            unique_labels = label_values
        else:
            unique_labels = np.unique(original_data)
            unique_labels = unique_labels[unique_labels != 0]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for label_val in tqdm(unique_labels, desc="Processing labels", disable=not show_time):
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
                #testing break
                # break
        
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting results", disable=not show_time):
                try:
                    thinned_data = future.result()
                    label_list.append(thinned_data.astype(np.uint8))
                    # midline_labels += thinned_data.astype(np.uint8)
                except Exception as e:
                    print(f"Error processing a label: {e}")

        #TODO, check for volumetric label intersections and fix them here
        # midline_labels = check_for_intersections(label_list)
        midline_labels = np.stack(midline_labels, axis=0)
        if show_time:
            print(f"Total time taken to process all structures: {time.time() - overall_start_time:.2f} seconds")
        
        nrrd.write(output_path, midline_labels.astype(np.uint8), header)
        return midline_labels
    except Exception as e:
        print(f"Unexpected error processing file {nrrd_path}: {e}")
        return None

def process_directory(input_directory, pad_amount=0, label_values=None, test_mode=False, replace=False, show_time=False, filter_labels=False):
    processed_count = 0
    files_to_process = []
    
    for root, _, files in os.walk(input_directory):
        mask_file = next((f for f in files if f.endswith('_mask.nrrd')), None)
        if mask_file:
            input_path = os.path.join(root, mask_file)
            output_path = input_path.replace('_mask.nrrd', '_graph_mask_thinned.nrrd')
            
            if not os.path.exists(output_path) or replace:
                files_to_process.append((input_path, output_path))
    
    for input_path, output_path in tqdm(files_to_process, desc="Processing files"):
        if show_time:
            overall_start_time = time.time()
        
        result = process_structures(input_path, output_path, pad_amount, label_values, show_time=show_time, filter_labels=filter_labels)
        
        if result is None:
            print(f"Failed to process file {os.path.basename(input_path)}")
        elif show_time:
            print(f"Total time taken for file {os.path.basename(input_path)}: {time.time() - overall_start_time:.2f} seconds")
        
        processed_count += 1
        if test_mode and processed_count >= 1:
            if show_time:
                print(f"Test mode: Processed {processed_count} files, stopping.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D instance volumes to midline volumes.')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only first 2 labels)')
    parser.add_argument('--replace', action='store_true', help='Replace existing _graph_mask_thinned.nrrd files')
    parser.add_argument('--time', action='store_true', help='Show timing information and progress bars')
    parser.add_argument('--filter-labels', action='store_true', help='Filter and reassign labels')
    args = parser.parse_args()

    current_directory = os.getcwd()
    input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    process_directory(input_directory, pad_amount=0, label_values=label_values, test_mode=args.test, replace=args.replace, show_time=args.time, filter_labels=args.filter_labels)
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")