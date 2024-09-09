import os
import re
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from midline_helper_simplified import create_slicer_nrrd_header
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
This script is used to relabel the volumetric cube labels so that the same sheets
have the same label values across different cubes. 
"""

def find_mask_files(input_directory):
    mask_files = {}
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('_mask.nrrd'):
                # Extract z_y_x from the filename
                match = re.search(r'(\d+_\d+_\d+)_mask\.nrrd', file)
                if match:
                    z_y_x = match.group(1)
                    file_path = os.path.join(root, file)
                    mask_files[z_y_x] = file_path
    
    return mask_files

def group_labels(mask_files, cube_size):
    label_groups = {}
    processed_cubes = set()
    next_group_id = 1

    def merge_groups(group1_id, group2_id):
        nonlocal label_groups
        label_groups[group1_id].update(label_groups[group2_id])
        del label_groups[group2_id]

    def find_group(item):
        for group_id, group in label_groups.items():
            if item in group:
                return group_id
        return None

    def process_cube(z_y_x):
        nonlocal next_group_id
        if z_y_x in processed_cubes:
            return

        cube_data, cube_header = nrrd.read(mask_files[z_y_x])
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
                adj_cube_data, adj_header = nrrd.read(mask_files[adj_z_y_x])
                connected_labels = is_connected(cube_data, cube_header, adj_cube_data, adj_header, cube_size=cube_size)
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
            process_cube(z_y_x)

    return label_groups

def is_connected(cube1, header1, cube2, header2, cube_size=256, vis=False):
    # Extract space origin from headers
    origin1 = np.array(header1['space origin'])
    origin2 = np.array(header2['space origin'])

    # Calculate the difference to determine which face is adjacent
    diff = origin2 - origin1

    # Determine the axis and index of the adjacent face
    axis = np.argmax(np.abs(diff))
    is_positive = diff[axis] > 0

    # Check if faces are actually adjacent using cube_size
    if np.abs(diff[axis]) != cube_size or np.any(np.abs(diff[np.arange(3) != axis]) > 1e-6):
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
        if best_match is not None:
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

#Buggy; removing labels etc on cube 2
def relabel_paired_cubes(cube1, header1, p1, cube2, header2, p2, paired_labels, output_folder):
    # Create a mapping for the second cube
    label_map = dict(paired_labels)  # Use paired_labels directly
    # print(label_map)

    os.makedirs(output_folder, exist_ok=True)
    # Find the highest label value in both cubes
    max_label = max(np.max(cube1), np.max(cube2))
    
    # Create a new array for the relabeled cube2
    relabeled_cube2 = np.zeros_like(cube2)

    for new_label, old_label in label_map.items():
        # If the old_label is in cube2, assign its area to relabeled_cube2 with the new_label value
        if old_label in np.unique(cube2):
            relabeled_cube2[cube2 == old_label] = new_label
            # Set the assigned area to 0 in cube2
            cube2[cube2 == old_label] = 0
    
    # Assign remaining non-zero labels in cube2 to new unique labels
    for label in np.unique(cube2):
        if label != 0:
            max_label += 1
            relabeled_cube2[cube2 == label] = max_label
    
    # Save relabeled cubes
    for cube, header, original_path in [(cube1, header1, p1), (relabeled_cube2, header2, p2)]:
        filename = os.path.basename(original_path)
        new_filename = filename.replace('_mask.nrrd', f'_relabeled_mask.nrrd')
        output_path = os.path.join(output_folder, new_filename)
        z,y,x = header['space origin']
        print(z,y,x)
        header2 = create_slicer_nrrd_header(cube,z,y,x)
        nrrd.write(output_path, cube, header2)
    
    return relabeled_cube2, label_map

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

def relabel_single_cube(z_y_x, file_path, output_folder, label_group_mapping):
        cube, header = nrrd.read(file_path)
        dtype = np.uint16
        cube = cube.astype(dtype)
        relabeled_cube = np.zeros_like(cube, dtype=dtype)
        
        # Create a mapping for this cube
        label_map = {label: group_id for label, group_id in label_group_mapping[z_y_x]}
        # print(label_map)
        # Relabel the cube
        for old_label, new_label in label_map.items():
            relabeled_cube[cube == old_label] = new_label
        
        # Handle any labels not in the mapping, should be none
        unmapped_labels = set(np.unique(cube)) - set(label_map.keys()) - {0}
        max_label = max(max(label_map.values()), np.max(relabeled_cube))
        for old_label in unmapped_labels:
            max_label += 1
            relabeled_cube[cube == old_label] = max_label
        
        # Save relabeled cube
        filename = os.path.basename(file_path)
        new_filename = filename.replace('_mask.nrrd', '_relabeled_mask.nrrd')
        output_path = os.path.join(output_folder, new_filename)
        z,y,x = z_y_x.split('_')
        z, y, x = int(z), int(y), int(x)
        new_header = create_slicer_nrrd_header(relabeled_cube, z, y, x, encoding='gzip')
        nrrd.write(output_path, relabeled_cube.astype(dtype), new_header)
        
        return f"Relabeled cube saved: {output_path}"

def relabel_cubes(label_group_mapping, mask_files, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(relabel_single_cube, z_y_x, file_path, output_folder, label_group_mapping) for z_y_x, file_path in mask_files.items()]
        for future in tqdm(as_completed(futures), total=len(mask_files), desc="Relabeling cubes"):
            # print(future.result())
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process cube labels and optionally visualize results.')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--cube-size', type=int, default=256, help='Cube size (default: 256)')
    args = parser.parse_args()

    current_directory = os.getcwd()
    input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    cube_size = args.cube_size

    mask_files = find_mask_files(input_directory)
    p1 = mask_files['01744_02256_04048']
    p2 = mask_files['01744_02256_04304']
    p3 = mask_files['01744_02256_04560']
    p4 = mask_files['01744_02512_04304']
    p5 = mask_files['01744_02000_04304']
    p6 = mask_files['01744_02000_04560']
    p8 = mask_files['01744_02512_04560']
    p9 = mask_files['01744_02512_04048']

    mask_files = {k: v for k, v in mask_files.items() if k in ['01744_02256_04048', '01744_02256_04304', '01744_02256_04560', '01744_02512_04304', '01744_02000_04304', '01744_02000_04560', '01744_02512_04560', '01744_02512_04048']}
    # mask_files = {k: v for k, v in mask_files.items() if k in ['00000_02408_04560', '00064_02664_04304']}
    
    # cube1, header1 = nrrd.read(p1)
    # cube2, header2 = nrrd.read(p2)
    # cube3, header3 = nrrd.read(p3)

    # result1 = is_connected(cube1, header1, cube2, header2, cube_size=cube_size, vis=args.vis)
    # result2 = is_connected(cube2, header2, cube3, header3, cube_size=cube_size, vis=args.vis)
    # result3 = is_connected(cube1, header1, cube3, header3, cube_size=cube_size, vis=args.vis)
    # print(result1)
    # print(result2)
    # print(result3)

    # relabel_paired_cubes(cube1, header1, p1, cube2, header2, p2, result1, output_folder=current_directory+'/relabeled_cubes')
    # # Print the first 3 results
    # for z_y_x, file_path in list(mask_files.items())[:3]:
    #     print(f"{z_y_x}: {file_path}")

    # Usage
    label_groups = group_labels(mask_files, cube_size)
    label_group_mapping = create_label_group_mapping(label_groups, mask_files)
    relabel_cubes(label_group_mapping, mask_files, output_folder=current_directory+'/relabeled_cubes/test2')
