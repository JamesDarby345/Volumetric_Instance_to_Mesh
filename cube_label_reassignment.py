import os
import re
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import argparse
from midline_helper_simplified import create_slicer_nrrd_header

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

def reassign_labels(mask_files, cube_size):
    label_value_map = {}
    processed_cubes = set()

    def process_cube(z_y_x):
        if z_y_x in processed_cubes:
            return

        cube_data, _ = nrrd.read(mask_files[z_y_x])
        unique_labels = np.unique(cube_data)

        for label in unique_labels:
            if label == 0:  # Assuming 0 is background
                continue
            if label not in label_value_map:
                label_value_map[label] = {z_y_x: label}
            else:
                label_value_map[label][z_y_x] = label

        processed_cubes.add(z_y_x)

        # Check adjacent cubes
        z, y, x = map(int, z_y_x.split('_'))
        adjacent_coords = [
            f"{z+cube_size}_{y}_{x}", f"{z-cube_size}_{y}_{x}",
            f"{z}_{y+cube_size}_{x}", f"{z}_{y-cube_size}_{x}",
            f"{z}_{y}_{x+cube_size}", f"{z}_{y}_{x-cube_size}"
        ]

        for adj_z_y_x in adjacent_coords:
            if adj_z_y_x in mask_files and adj_z_y_x not in processed_cubes:
                adj_cube_data, _ = nrrd.read(mask_files[adj_z_y_x])
                connected_labels = is_connected(cube_data, adj_cube_data)

                for label1, label2 in connected_labels:
                    if label1 in label_value_map:
                        if label2 not in label_value_map:
                            label_value_map[label1][adj_z_y_x] = label2
                        else:
                            # Merge label maps if both labels exist
                            label_value_map[label1].update(label_value_map[label2])
                            del label_value_map[label2]

                process_cube(adj_z_y_x)

    # Start processing from the first cube
    first_z_y_x = next(iter(mask_files))
    process_cube(first_z_y_x)

    return label_value_map


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
def relabel_cube(cube1, header1, p1, cube2, header2, p2, paired_labels, output_folder):
    # Create a mapping for the second cube
    label_map = dict(paired_labels)  # Use paired_labels directly
    print(label_map)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process cube labels and optionally visualize results.')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--cube-size', type=int, default=256, help='Cube size (default: 256)')
    args = parser.parse_args()

    current_directory = os.getcwd()
    input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    cube_size = args.cube_size

    mask_files = find_mask_files(input_directory)

    p1 = mask_files['02000_02000_02000']
    p2 = mask_files['02000_02256_02000']
    cube1, header1 = nrrd.read(p1)
    cube2, header2 = nrrd.read(p2)

    result = is_connected(cube1, header1, cube2, header2, cube_size=cube_size, vis=args.vis)
    print(result)

    relabel_cube(cube1, header1, p1, cube2, header2, p2, result, output_folder=current_directory+'/relabeled_cubes')
    # # Print the first 3 results
    # for z_y_x, file_path in list(mask_files.items())[:3]:
    #     print(f"{z_y_x}: {file_path}")

    # # Usage
    # label_value_map = reassign_labels(mask_files, cube_size)
