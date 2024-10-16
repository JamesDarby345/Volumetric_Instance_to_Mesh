import os
from matplotlib import pyplot as plt
import numpy as np
import nrrd
import time
import concurrent.futures
import argparse
from tqdm import tqdm
from scipy.spatial import  KDTree
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import pyvista as pv
# from scipy.ndimage import binary_erosion
import umap
from numba import jit
# import fastmorph

from midline_helper_simplified import *

"""
The goal of this file is to load in harmonised nrrd cube volumatric instance label files
and then for each instance, find the midline sheet/ manifoldof the instance across cubes.

We will do this by representing each instance, using a kdtree to iterate over local neighbourhoods
where the assumption that the instance can be represented by a single voxelised value intersecting the normal
of the surface holds.

the kdtree local neighbourhood search can then also be used to create a mesh via pyvista's 
delauney_2d over the local neighbourhoods.
"""

# @jit(nopython=True)
# def get_voxel_coords(volume, label):
#     coords = []
#     it = np.nditer(volume, flags=['multi_index'])
#     while not it.finished:
#         if it[0] == label:
#             coords.append(it.multi_index)
#         it.iternext()
#     return np.array(coords)

def visualize_point_clouds(point_clouds):
    plotter = pv.Plotter()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(point_clouds)))

    for i, (label, point_cloud) in enumerate(point_clouds.items()):
        # Create a PyVista point cloud
        pv_point_cloud = pv.PolyData(point_cloud)

        color = colors[i][:3]  # RGB values
        # Add the point cloud to the plotter with a unique color
        plotter.add_mesh(
            pv_point_cloud,
            point_size=10.0,
            render_points_as_spheres=True,
            color=color,
            label=f"Label {label}"
        )

    plotter.add_legend()
    plotter.show_axes()
    plotter.show()

def generate_seed_points(coords, radius, tree):
    """
    Generate seed points to cover the structure using a simple greedy algorithm.
    """
    remaining = set(range(len(coords)))
    seed_points = []
    while remaining:
        idx = remaining.pop()
        seed_point = coords[idx]
        seed_points.append(seed_point)
        # Find neighbors within radius
        neighbors = tree.query_ball_point(seed_point, r=radius)
        remaining -= set(neighbors)
    return np.array(seed_points)

def thin_structure(coords, normals, seed_points, radius, tree, resolution='sparse', sample_size=10):
    """
    ----< slow > Thins the structure by projecting points onto planes perpendicular to the normals in the local neighbourhood. < slow >----
    Thins the structure by averaging the points in the local neighbourhood.
    """

    thinned_points = []
    for i, seed in enumerate(seed_points):
        # Find points within the neighborhood
        idx = tree.query_ball_point(seed, r=radius)
        neighborhood = coords[idx]
        normal = normals[i]

        # Project points onto the plane
        mean_point = neighborhood.mean(axis=0)
        centered = neighborhood - mean_point
        
        if resolution == 'sparse':
            thinned_points.append(mean_point)
        elif resolution == 'voxelise':
            return coords
        elif resolution == 'dense':
            projected = centered - np.dot(centered, normal[:, np.newaxis]) * normal
            thinned_points.extend(projected + mean_point)
        else:
            projected = centered - np.dot(centered, normal[:, np.newaxis]) * normal
            # Subsample the projected points
            num_points = len(projected)
            subsample_size = min(sample_size, num_points)  # Take up to sample_size points or all if less
            subsample_indices = np.random.choice(num_points, subsample_size, replace=False)
            subsampled_points = projected[subsample_indices] + mean_point
            thinned_points.extend(subsampled_points)

    return np.array(thinned_points)

def process_single_file_to_point_clouds(paths, radius, resolution='sparse', morph_radius=10):
    input_path, output_path = paths
    zyx = os.path.basename(os.path.dirname(input_path)).split('_')
    zyx = [int(coord) for coord in zyx]
    point_clouds = {}  # Dictionary to store point clouds for each label

    try:
        # Load the input NRRD file
        data, header = nrrd.read(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None

    unique_labels = np.unique(data)
    for label in unique_labels:
        if label == 0:
            continue
        volume = np.where(data == label, label, 0)
        if morph_radius > 0:
            volume = morphological_tunnel_filling(volume, label, morph_radius)
        coords = np.array(np.where(volume == label)).T  # Shape (N, 3)
        # Add zyx values to each entry in coords
        zyx_array = np.array(zyx)
        coords = coords + zyx_array
        coords = coords.astype(np.float32)
        # Build KD-tree
        tree = KDTree(coords)

        # Generate seed points
        seed_points = generate_seed_points(coords, radius, tree)

        # Compute normals using PCA for each neighborhood
        normals = []
        for seed in seed_points:
            idx = tree.query_ball_point(seed, r=radius)
            neighborhood = coords[idx]
            if len(neighborhood) >= 3:
                pca = PCA(n_components=3)
                pca.fit(neighborhood)
                normal = pca.components_[-1]  # The normal vector
            else:
                normal = np.array([0, 0, 1])  # Default normal if not enough points
            normals.append(normal)
        normals = np.array(normals)

        # Thin the structure
        thinned_points = thin_structure(coords, normals, seed_points, radius, tree, resolution)
       
        # Store the point cloud for this label
        point_clouds[label] = pv.PolyData(thinned_points)   

    # Visualization with PyVista for all labels
    # if point_clouds:
    #     visualize_point_clouds(point_clouds)

    return point_clouds
                
def process_directory(input_directory, output_directory, replace=False, test_cubes=[], show_time=False, radius=20.0, resolution='sparse', morph_radius=10):
    files_to_process = []
    for root, _, files in os.walk(input_directory):
        mask_file_suffix = '_relabeled_mask.nrrd'
        mask_file = next((f for f in files if f.endswith(mask_file_suffix)), None)
        if mask_file:
            input_path = os.path.join(root, mask_file)
            output_path = input_path.replace(mask_file_suffix, '_kdtree_mask_thinned.nrrd')
            if os.path.exists(output_path) and not replace:
                continue
            files_to_process.append((input_path, output_path))
    if len(test_cubes) > 0:
        # Filter files_to_process based on test_cubes
        files_to_process = [
            (input_path, output_path) 
            for input_path, output_path in files_to_process 
            if any(cube in input_path for cube in test_cubes)
        ]

    point_clouds = {}

    # Process files in parallel, creating a dictionary of label values and their associated thinned point clouds
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file_to_point_clouds, args, radius, resolution, morph_radius) for args in files_to_process]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_process), desc="Processing files", unit="file"):
            result = future.result()
            if result:
                for label, point_cloud in result.items():
                    if label in point_clouds:
                        if isinstance(point_clouds[label], list):
                            point_clouds[label].append(point_cloud)
                        else:
                            point_clouds[label] = [point_clouds[label], point_cloud]
                    else:
                        point_clouds[label] = point_cloud
    
    # Merge point clouds for each label
    merged_point_clouds = {}
    for label, clouds in point_clouds.items():
        if isinstance(clouds, list):
            # If multiple point clouds, merge them
            merged_cloud = clouds[0]
            for cloud in clouds[1:]:
                merged_cloud = merged_cloud.merge(cloud)
        else:
            # If single point cloud, use as is
            merged_cloud = clouds
        merged_point_clouds[label] = merged_cloud

    # Save the merged point clouds to the output directory
    os.makedirs(output_directory, exist_ok=True)
    
    for label, point_cloud in merged_point_clouds.items():
        # Isomap dimensionality reduction from 3d point cloud to 2d
        # n_neighbours = len(point_cloud.points) // 100
        n_neighbours = int(100 // radius)
        if resolution != 'sparse':
            n_neighbours = int(300 // radius)
        n_neighbours = max(n_neighbours, 3)
        # n_neighbours = min(n_neighbours, len(point_cloud.points) - 1)

        isomap = Isomap(n_components=2, n_neighbors=n_neighbours)
        reduced_coords = isomap.fit_transform(point_cloud.points)

        # reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbours)
        # reduced_coords = reducer.fit_transform(point_cloud.points)

        # Create a new point cloud object with reduced dimensions
        reduced_point_cloud = pv.PolyData(np.column_stack((reduced_coords, np.zeros(len(reduced_coords)))))
        print(len(reduced_point_cloud.points), len(point_cloud.points))

        # Apply Delaunay triangulation to the reduced point cloud
        delaunay = reduced_point_cloud.delaunay_2d()

        # Get the faces from the Delaunay triangulation
        # faces = delaunay.faces.reshape(-1, 4)[:, 1:]

        # Create a new mesh using the original 3D points and the faces from Delaunay
        mesh = pv.PolyData(point_cloud.points, delaunay.faces)
        mesh.compute_normals(auto_orient_normals=True, inplace=True)

        # Save the mesh
        output_path = os.path.join(output_directory, f'{label}_mesh.obj')
        mesh.save(output_path)

        output_path = os.path.join(output_directory, f'{label}_point_cloud.obj')
        point_cloud.save(output_path)

        output_path = os.path.join(output_directory, f'{label}_reduced_point_cloud.obj')
        reduced_point_cloud.save(output_path)
    
    # return merged_point_clouds
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D instance volumes to point clouds.')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--replace', action='store_true', help='Replace existing _mask_thinned.nrrd files')
    parser.add_argument('--time', action='store_true', help='Show execution time for each file')
    parser.add_argument('--input-dir', type=str, help='Input directory path')
    parser.add_argument("--output-path", type=str, help="Path to the directory to save point clouds into")
    parser.add_argument("--radius", type=float, default=20.0, help="Radius for point cloud generation")
    parser.add_argument("--resolution", type=str, default='sparse', help="Resolution for point cloud generation; sparse, dense or subsample")
    parser.add_argument("--morph", type=int, default=0, help="Radius for morphological tunnel filling, 30 is good; adds signifigant time to processing")
    args = parser.parse_args()

    current_directory = os.getcwd()
    default_input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'
    input_directory = args.input_dir if args.input_dir else default_input_directory
    output_directory = args.output_path if args.output_path else os.path.join(current_directory, 'point_clouds')
    label_values = None  # List of label values to process, pass None to process all labels
    overall_start_time = time.time()
    test_cubes= []
    if args.test:
        test_cubes = ['01744_02256_03792','01744_02256_04048', '01744_02256_04304', '01744_02000_04048', '01744_02000_04304']

    process_directory(input_directory, output_directory, replace=args.replace, test_cubes=test_cubes, show_time=args.time, radius=args.radius, resolution=args.resolution, morph_radius=args.morph)
    
    if args.time:
        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")