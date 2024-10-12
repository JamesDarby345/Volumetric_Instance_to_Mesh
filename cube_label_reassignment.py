import os
import re
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import argparse
import json

from tqdm import tqdm
from midline_helper_simplified import create_slicer_nrrd_header, generate_light_colormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from filter_and_reassign_labels import process_mask_files

"""
This script is used to relabel the volumetric cube labels so that the same sheets
have the same label values across different cubes. 
"""

def find_mask_files(input_directory, suffix=None, test_cubes=None):
    mask_files = {}
    if not suffix:
        suffix = '_mask.nrrd'
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(suffix):
                # Extract z_y_x from the filename
                match = re.search(r'(\d+_\d+_\d+)' + suffix, file)
                if match:
                    z_y_x = match.group(1)
                    if test_cubes is None or z_y_x in test_cubes:
                        file_path = os.path.join(root, file)
                        mask_files[z_y_x] = file_path
    
    return mask_files

def group_labels(mask_files, cube_size, half_winding_plane=None, overlap_min=300):
    label_groups = {}
    processed_cubes = set()
    next_group_id = 1
    errors = []

    def merge_groups(group1_id, group2_id):
        nonlocal label_groups
        label_groups[group1_id].update(label_groups[group2_id])
        del label_groups[group2_id]

    def find_group(item):
        for group_id, group in label_groups.items():
            if item in group:
                return group_id
        return None

    def process_cube(z_y_x, overlap_min=300):
        nonlocal next_group_id
        if z_y_x in processed_cubes:
            return

        try:
            cube_data, cube_header = nrrd.read(mask_files[z_y_x])
            # Check if the dimension is 3
            if cube_header.get('dimension') != 3:
                return f"Error: Cube {z_y_x} has incorrect dimension: {cube_header.get('dimension')}. Expected 3."
        except Exception as e:
            errors.append((z_y_x, str(e)))
            return

        processed_cubes.add(z_y_x)

        # Check adjacent cubes
        z, y, x = map(int, z_y_x.split('_'))
        adjacent_coords = [
            f"{str(z+cube_size).zfill(5)}_{str(y).zfill(5)}_{str(x).zfill(5)}",
            f"{str(z-cube_size).zfill(5)}_{str(y).zfill(5)}_{str(x).zfill(5)}",
            f"{str(z).zfill(5)}_{str(y+cube_size).zfill(5)}_{str(x).zfill(5)}",
            f"{str(z).zfill(5)}_{str(y-cube_size).zfill(5)}_{str(x).zfill(5)}",
            f"{str(z).zfill(5)}_{str(y).zfill(5)}_{str(x+cube_size).zfill(5)}",
            f"{str(z).zfill(5)}_{str(y).zfill(5)}_{str(x-cube_size).zfill(5)}"
        ]

        for adj_z_y_x in adjacent_coords:
            if adj_z_y_x in mask_files and adj_z_y_x not in processed_cubes:
                try:
                    adj_cube_data, adj_header = nrrd.read(mask_files[adj_z_y_x])
                except Exception as e:
                    errors.append((adj_z_y_x, str(e)))
                    continue
                
                # Check if the connection crosses the half-winding plane
                if half_winding_plane:
                    axis, value = half_winding_plane
                    axis_index = {'x': 2, 'y': 1, 'z': 0}[axis]
                    current_value = int(z_y_x.split('_')[axis_index])
                    adjacent_value = int(adj_z_y_x.split('_')[axis_index])
                    if (current_value < value <= adjacent_value) or (adjacent_value < value <= current_value):
                        continue  # Skip this connection if it crosses the half-winding plane
                
                connected_labels = is_connected(cube_data, cube_header, adj_cube_data, adj_header, cube_size=cube_size, overlap_min=overlap_min)
                print(z_y_x, adj_z_y_x, connected_labels)
                for label1, label2 in connected_labels:
                    item1 = (z_y_x, label1)
                    item2 = (adj_z_y_x, label2)
                    group1_id = find_group(item1)
                    group2_id = find_group(item2)

                    if group1_id is None and group2_id is None:
                        label_groups[next_group_id] = {item1, item2}
                        next_group_id += 1
                    elif group1_id is None:
                        label_groups[group2_id].add(item1)
                    elif group2_id is None:
                        label_groups[group1_id].add(item2)
                    elif group1_id != group2_id:
                        merge_groups(group1_id, group2_id)

        # Add any remaining labels in the current cube to new groups
        for label in np.unique(cube_data):
            if label == 0:  # Skip background
                continue
            item = (z_y_x, label)
            if find_group(item) is None:
                label_groups[next_group_id] = {item}
                next_group_id += 1

    # Process all cubes
    # Could parralelise this, but would have to sync changes to label_groups, so non-trivial and its pretty fast anyway
    # If wanting to run on large amounts of cubes, would need to do this, and probably use a half winding approach to prevent
    # all the volumetric labels connecting and being assigned the same label
    for z_y_x in mask_files:
        if z_y_x not in processed_cubes:
            process_cube(z_y_x, overlap_min=overlap_min)

    # Print errors at the end
    if errors:
        print("\nErrors encountered while processing cubes:")
        for cube, error in errors:
            print(f"Cube {cube}: {error}")

    return label_groups

def is_connected(cube1, header1, cube2, header2, cube_size=256, vis=False, overlap_min=300):
    # Extract space origin from headers
    origin1 = np.array(header1['space origin'])
    origin2 = np.array(header2['space origin'])
    # print(origin1, origin2)
    # Calculate the difference to determine which face is adjacent
    diff = origin2 - origin1

    # Determine the axis and index of the adjacent face
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    # Check if faces are actually adjacent using cube_size
    if np.abs(diff[axis]) != cube_size or np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6):
        print('no faces adjacent')
        print('np.abs(diff[axis]) != cube_size ', np.abs(diff[axis]) != cube_size, np.abs(diff[axis]))
        print('np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6) ', np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6))
        return []  # No faces are adjacent, return an empty list

    face_index = -1 if is_positive else 0

    # Extract the adjacent faces
    if axis == 0:
        face1 = cube1[face_index, :, :]
        face2 = cube2[-face_index-1, :, :]
    elif axis == 1:
        face1 = cube1[:, face_index, :]
        face2 = cube2[:, -face_index-1, :]
    else:
        face1 = cube1[:, :, face_index]
        face2 = cube2[:, :, -face_index-1]

    # Visualization code wrapped in conditional blocks
    if vis:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot face1
        im1 = ax1.imshow(face1, cmap='viridis')
        ax1.set_title('Face from Cube 1')
        plt.colorbar(im1, ax=ax1)

        # Plot face2
        im2 = ax2.imshow(face2, cmap='viridis')
        ax2.set_title('Face from Cube 2')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()

    # Find connected labels
    unique_labels1 = np.unique(face1)
    unique_labels2 = np.unique(face2)
    connected_labels = []

    for label1 in unique_labels1:
        if label1 == 0:  # Skip background
            continue
        mask1 = face1 == label1
        max_overlap = 0
        best_match = None
        for label2 in unique_labels2:
            if label2 == 0:  # Skip background
                continue
            mask2 = face2 == label2
            overlap = np.sum(mask1 & mask2)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = label2
        if best_match is not None and max_overlap > overlap_min:
            connected_labels.append((label1, best_match))

    # Create a new array for the recolored face2
    recolored_face2 = np.copy(face2)

    # Recolor face2 labels with matching face1 values
    for label1, label2 in connected_labels:
        recolored_face2[face2 == label2] = label1

    if vis:
        # Function to display masked results side by side
        def display_masked_faces(face1, mask1, face2, mask2, label):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            masked_face1 = np.ma.masked_where(mask1 == 0, face1)
            masked_face2 = np.ma.masked_where(mask2 == 0, face2)
            
            im1 = ax1.imshow(masked_face1, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'Face 1 - Label {label}')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(masked_face2, cmap='viridis', interpolation='nearest')
            ax2.set_title(f'Recolored Face 2 - Label {label}')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.show()

        # Go through each matching value in the recolored face1 and face2
        for label1, label2 in connected_labels:
            mask1 = face1 == label1
            mask2 = recolored_face2 == label1  # Use label1 here since face2 has been recolored

            display_masked_faces(face1, mask1, recolored_face2, mask2, label1)

    # Find labels that don't have a match in face1 and recolored face2
    unmatched_labels_face1 = set(np.unique(face1)) - {0} - set(label1 for label1, _ in connected_labels)
    unmatched_labels_face2 = set(np.unique(face2)) - {0} - set(label2 for _, label2 in connected_labels)

    # Create masks for all unmatched labels
    mask1_unmatched = np.isin(face1, list(unmatched_labels_face1))
    mask2_unmatched = np.isin(face2, list(unmatched_labels_face2))

    # Display all unmatched labels at once
    if vis:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        masked_face1 = np.ma.masked_where(~mask1_unmatched, face1)
        masked_face2 = np.ma.masked_where(~mask2_unmatched, face2)

        im1 = ax1.imshow(masked_face1, cmap='viridis', interpolation='nearest')
        ax1.set_title(f'Unmatched in Face 1: {unmatched_labels_face1}')
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(masked_face2, cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Unmatched in Face 2: {unmatched_labels_face2}')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.show()

    return connected_labels

def create_label_group_mapping(label_groups, mask_files):
    mapping = {}
    
    for z_y_x in mask_files:
        mapping[z_y_x] = []
        for group_id, group in label_groups.items():
            for cube_z_y_x, label in group:
                if cube_z_y_x == z_y_x:
                    mapping[z_y_x].append((label, group_id))
        
        # Sort the list by label value
        mapping[z_y_x].sort(key=lambda x: x[0])
    
    return mapping

def relabel_single_cube(z_y_x, colors, file_path, output_folder, label_group_mapping, overwrite, downsample_factor, dsonly):
    try:
        cube, header = nrrd.read(file_path)
        
        if header.get('dimension') != 3:
            return f"Error: Cube {z_y_x} has incorrect dimension: {header.get('dimension')}. Expected 3."
        
    except Exception as e:
        return f"Error reading cube {z_y_x}: {str(e)}"

    if output_folder is None:
        output_folder = os.path.dirname(file_path)

    dtype = np.uint16
    cube = cube.astype(dtype)
    relabeled_cube = np.zeros_like(cube, dtype=dtype)
    
    label_map = {label: group_id for label, group_id in label_group_mapping[z_y_x]}
    for old_label, new_label in label_map.items():
        relabeled_cube[cube == old_label] = new_label
    
    unmapped_labels = set(np.unique(cube)) - set(label_map.keys()) - {0}
    max_label = max(max(label_map.values()), np.max(relabeled_cube))
    for old_label in unmapped_labels:
        max_label += 1
        relabeled_cube[cube == old_label] = max_label
    
    filename = os.path.basename(file_path)
    if not overwrite:
        new_filename = filename.replace('_mask.nrrd', '_relabeled_mask.nrrd')
    else:
        new_filename = filename
    output_path = os.path.join(output_folder, new_filename)
    z, y, x = z_y_x.split('_')
    z, y, x = int(z), int(y), int(x)
    
    if downsample_factor:
        relabeled_cube = relabeled_cube[::downsample_factor, ::downsample_factor, ::downsample_factor]
        ds_z, ds_y, ds_x = z // downsample_factor, y // downsample_factor, x // downsample_factor
        new_filename = new_filename.replace('_mask.nrrd', f'_d{downsample_factor}_mask.nrrd')
        ds_output_path = os.path.join(output_folder, new_filename)
        ds_header = create_slicer_nrrd_header(relabeled_cube, colors, ds_z, ds_y, ds_x, encoding='gzip')
        nrrd.write(ds_output_path, relabeled_cube.astype(dtype), ds_header)
        if dsonly:
            return f"Downsampled only relabeled cube saved: {ds_output_path}"
    
    new_header = create_slicer_nrrd_header(relabeled_cube, colors, z, y, x, encoding='gzip')
    nrrd.write(output_path, relabeled_cube.astype(dtype), new_header)
    
    return f"Relabeled cube saved: {output_path}"

def relabel_cubes(label_group_mapping, mask_files, colors, output_folder=None, overwrite=False, downsample_factor=None, dsonly=False):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    errors = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(relabel_single_cube, z_y_x, colors, file_path, output_folder, label_group_mapping, overwrite, downsample_factor, dsonly) 
                   for z_y_x, file_path in mask_files.items()]
        for future in tqdm(as_completed(futures), total=len(mask_files), desc="Relabeling cubes"):
            result = future.result()
            if result.startswith("Error"):
                errors.append(result)

    if errors:
        print("\nErrors encountered while relabeling cubes:")
        for error in errors:
            print(error)

def uint8_to_int(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: uint8_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [uint8_to_int(item) for item in obj]
    return obj

def save_label_groups_to_json(label_groups):
    
    output_path = 'label_groups.json'

    data_to_save = []
    for group_id, groups in label_groups.items():
        # label_list = []
        zyx_list = []
        for zyx_str, label_val in groups:
            # label_list.append(label_val)
            zyx_list.append(zyx_str)
        data_to_save.append({
            'harmonised_label_value': group_id,
            # 'label_values': label_list,
            'zyx_values': zyx_list
        })

    # Convert uint8 to int
    data_to_save = uint8_to_int(data_to_save)

    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Relabel sheets across nrrd cubes.')
    parser.add_argument('--vis', action='store_true', help='Enable visualization for testing')
    parser.add_argument('--cube-size', type=int, default=256, help='Cube size (default: 256)')
    parser.add_argument("--input-path", type=str, help="Path to the directory containing mask files")
    parser.add_argument("--output-path", type=str, help="(Optional) Path to the directory to save relabeled mask files into")
    parser.add_argument("--overwrite", action='store_true', help='Overwrite existing files instead of appending _relabeled_mask to the end of file names')
    parser.add_argument("--no-filter", action='store_true', help='Dont run filter_and_reassign_labels before relabeling if passed')
    parser.add_argument("--half-winding-plane", type=str, default=None, help='Specify the half-winding plane (e.g., "x:4048")')
    parser.add_argument("--suffix", type=str, default=None, help="Specify the suffix of the mask files (e.g., '_morphed_mask.nrrd')")
    parser.add_argument('--test', action='store_true', help='Run in test mode with a predefined set of cubes')
    parser.add_argument('--overlap-min', type=int, default=300, help='Minimum voxel overlap to consider two labels as connected')
    parser.add_argument('--downsample', type=int, default=None, choices=[2,4,8,16,32], help='Downsample factor (e.g., 2,4,8,16)')
    parser.add_argument('--dsonly', action='store_true', help='Save only downsampled files')
    args = parser.parse_args()

    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_path if args.input_path else default_input_directory
    output_directory = args.output_path if args.output_path else None

    # by default, run filter_and_reassign_labels on the input directory *_mask.nrrd files, but not *_relabeled_mask.nrrd files
    if not args.no_filter:
        process_mask_files(input_directory)

    if not args.suffix:
        suffix = '_mask.nrrd'
    else:
        suffix = args.suffix

    test_cubes = None
    if args.test:
        test_cubes = ['01744_02256_04048', '01744_02000_04048', '01744_02000_04304']
        # '01744_02256_03792', '01744_02256_04304',

    mask_files = find_mask_files(input_directory, suffix)

    # If in test mode, filter mask_files to only include test_cubes
    if test_cubes:
        mask_files = {k: v for k, v in mask_files.items() if k in test_cubes}

    # Parse the half-winding plane argument
    if args.half_winding_plane: 
        half_winding_axis, half_winding_value = args.half_winding_plane.split(':')
        half_winding_plane = (half_winding_axis, int(half_winding_value))
    else:
        half_winding_plane = None

    label_groups = group_labels(mask_files, args.cube_size, half_winding_plane=half_winding_plane, overlap_min=args.overlap_min )
    label_group_mapping = create_label_group_mapping(label_groups, mask_files)
    save_label_groups_to_json(label_groups)

    print(f'Generating colormap for {len(label_groups)} labels...')
    colors = generate_light_colormap(len(label_groups))
    relabel_cubes(label_group_mapping, mask_files, colors, output_folder=output_directory, overwrite=args.overwrite, downsample_factor=args.downsample, dsonly=args.dsonly)