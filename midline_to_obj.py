import sys
import numpy as np
from tqdm import tqdm
import nrrd
import os
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree
import multiprocessing
import concurrent.futures
import time
import glob
import argparse
from sklearn.decomposition import PCA
from midline_helper_simplified import *
import re

def print_timing(message, elapsed_time, should_print):
    if should_print:
        print(f"{message}: {elapsed_time:.4f} seconds")

def visualize_mesh(mesh):
    """
    Visualize the mesh using PyVista.

    Parameters:
    mesh (pyvista.PolyData): The mesh to visualize.
    """
    print("Visualizing mesh...")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='red', show_edges=True, opacity=0.7)
    plotter.add_points(mesh.points, color='blue', point_size=5)
    plotter.show_axes()
    plotter.show()

def calculate_average_pca_normal(original_array):
    """
    Calculate the average PCA normal across all sheets in the ROI.
    
    Parameters:
    original_array (numpy.ndarray): The original 3D array containing all sheets.
    
    Returns:
    numpy.ndarray: The average PCA normal vector.
    """
    indices = np.argwhere(original_array != 0)
    pca = PCA(n_components=3)
    pca.fit(indices)
    return pca.components_[2]  # The third component is normal to the main plane

def fix_mesh_normals(tm_mesh, avg_pca_normal, should_print_timing):
    start_time = time.time()
    """
    Fix mesh normals to point consistently towards the average PCA normal.
    
    Parameters:
    tm_mesh (trimesh.Trimesh): The input mesh.
    avg_pca_normal (numpy.ndarray): The average PCA normal vector.
    
    Returns:
    trimesh.Trimesh: The mesh with fixed normals.
    """
    # Calculate the average normal of the mesh
    avg_mesh_normal = np.mean(tm_mesh.face_normals, axis=0)
    
    # Check if the average mesh normal is pointing away from the PCA normal
    if np.dot(avg_mesh_normal, avg_pca_normal) < 0:
        tm_mesh.invert()
    
    end_time = time.time()
    print_timing("fix_mesh_normals", end_time - start_time, should_print_timing)
    return tm_mesh

def pyvista_to_trimesh(pv_mesh, avg_pca_normal, should_print_timing, should_fix_normals=True):
    """
    Convert a PyVista mesh to a Trimesh object, preserving UV coordinates and fixing normals.
    
    Parameters:
    pv_mesh (pyvista.PolyData): The input PyVista mesh.
    avg_pca_normal (numpy.ndarray): The average PCA normal vector.
    
    Returns:
    trimesh.Trimesh: The converted Trimesh object with fixed normals.
    """
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:4]
    
    # Create the Trimesh object
    tm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add UV coordinates if they exist
    if 'UV' in pv_mesh.point_data:
        uv_coords = pv_mesh.point_data['UV']
        tm_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords)

    if should_fix_normals:
        # Fix normals across sheets using the average PCA normal
        tm_mesh = fix_mesh_normals(tm_mesh, avg_pca_normal, should_print_timing)
    
    return tm_mesh

def add_uv_mapping(mesh):
    """
    Add UV coordinates to the mesh using a simple planar projection.
    
    Parameters:
    mesh (pyvista.PolyData): The input mesh.
    
    Returns:
    pyvista.PolyData: The mesh with UV coordinates added.
    """
    # Get the mesh points
    points = mesh.points
    
    # Normalize X and Y coordinates to [0, 1] range for UV mapping
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    uv_coords = (points[:, :2] - min_xy) / (max_xy - min_xy)
    
    # Add the UV coordinates to the mesh
    mesh.point_data['UV'] = uv_coords
    
    # print("Added UV mapping to the mesh.")
    return mesh

def filter_disconnected_parts(mesh, min_vertices):
    """
    Filter out disconnected parts of the mesh with fewer than min_vertices.
    
    Parameters:
    mesh (pyvista.PolyData): The input mesh.
    min_vertices (int): The minimum number of vertices a part should have to be kept.
    
    Returns:
    pyvista.PolyData: The filtered mesh.
    """
    # Get connected regions
    labeled = mesh.connectivity(largest=False)
    
    # Count vertices in each region
    unique_labels, counts = np.unique(labeled.cell_data['RegionId'], return_counts=True)
    
    # Create a mask for regions to keep
    keep_mask = np.isin(labeled.cell_data['RegionId'], unique_labels[counts >= min_vertices])
    
    # Extract the kept regions
    filtered_mesh = labeled.extract_cells(keep_mask)
    
    # print(f"Filtered out {len(unique_labels) - np.sum(counts >= min_vertices)} disconnected parts")
    # print(f"Remaining parts: {np.sum(counts >= min_vertices)}")
    
    return filtered_mesh

def array_to_thin_sheet_obj(array, filename, max_distance=1.8, min_vertices=800, space_origin=None, reconnection_mult=3, avg_pca_normal=None, should_print_timing=False, should_fix_normals=True):
    """
    Convert a 3D numpy array to a thin sheet-like mesh file, connecting only nearby voxels.
    
    Parameters:
    array (numpy.ndarray): 3D numpy array representing the structure.
    filename (str): Name of the output file.
    max_distance (float): Maximum distance for connecting voxels.
    min_vertices (int): Minimum number of vertices for a disconnected part to be kept.
    space_origin (tuple): The z, y, x space origin from the NRRD header.
    avg_pca_normal (numpy.ndarray): The average PCA normal vector.
    
    Returns:
    pyvista.PolyData: The resulting mesh.
    """
    start_time = time.time()
    
    # Find the indices of non-zero elements
    indices = np.argwhere(array != 0)
    
    # Create vertices from these indices
    vertices = indices.astype(float)
    
    # Apply space origin offset if provided
    if space_origin is not None:
        vertices += space_origin
    
    # print(f"Found {len(vertices)} non-zero elements.")
    
    if len(vertices) <= min_vertices:
        # print(f"Not enough points to create a mesh for {filename}, skipping.")
        return None

    prep_start = time.time()
    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(vertices)
    
    # Find pairs of points within max_distance
    pairs = tree.query_pairs(r=max_distance)
    
    # Create edges from these pairs
    edges = np.array(list(pairs))

    edges = np.hstack([[2, edge[0], edge[1]] for edge in edges])
    
    # Create a PyVista PolyData object
    mesh = pv.PolyData(vertices, lines=edges)
    
    # Convert lines to surface
    surf = mesh.delaunay_2d(alpha=max_distance)
    
    # Filter out disconnected parts
    surf = filter_disconnected_parts(surf, min_vertices=min_vertices)

    if surf.n_points == 0:
        print("No points left after filtering, skipping mesh creation.")
        return None
    
    surf = surf.delaunay_2d(alpha=max_distance*reconnection_mult)
    # Fill holes in the surface mesh
    surf = surf.fill_holes(hole_size=max_distance * 2 * reconnection_mult)
    
    #orient face normals in a consistent direction for each sheet, but not across sheets
    if should_fix_normals:
        surf.compute_normals(auto_orient_normals=True,inplace=True)
    surf = add_uv_mapping(surf)

    prep_end = time.time()
    print_timing("surface prep time", prep_end - prep_start, should_print_timing)
    
    # Convert to Trimesh
    trimesh_start = time.time()
    tm_mesh = pyvista_to_trimesh(surf, avg_pca_normal, should_print_timing, should_fix_normals=should_fix_normals)  # We don't need to fix normals in pyvista_to_trimesh anymore
    trimesh_end = time.time()
    print_timing("pyvista_to_trimesh", trimesh_end - trimesh_start, should_print_timing)
    
    # Save as OBJ
    with open(filename, 'w') as f:
        f.write("# OBJ file\n")
        for v in tm_mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        if tm_mesh.visual.uv is not None:
            for uv in tm_mesh.visual.uv:
                f.write(f"vt {uv[0]} {uv[1]}\n")
            
            for face in tm_mesh.faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
        else:
            for face in tm_mesh.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # print(f"OBJ file '{filename}' has been created.")
    
    end_time = time.time()
    print_timing("array_to_thin_sheet_obj", end_time - start_time, should_print_timing)
    return surf

def process_single_value(args):
    value, original_array, output_obj_path, max_distance, min_vertices, visualise, space_origin, reconnection_mult, avg_pca_normal, should_print_timing, should_fix_normals = args
    start_time = time.time()
    # print(f"Processing value {value}...")
    if np.sum(original_array == value) == 0:
        print(f"Value {value} not found in the input array, skipping.")
        return
    array = original_array.copy()
    array[original_array!=value] = 0
    temp_output_obj_path = f'{output_obj_path}_{value}.obj'
    mesh = array_to_thin_sheet_obj(array, temp_output_obj_path, max_distance=max_distance, min_vertices=min_vertices, space_origin=space_origin, reconnection_mult=reconnection_mult, avg_pca_normal=avg_pca_normal, should_print_timing=should_print_timing, should_fix_normals=should_fix_normals)
    if visualise:
        visualize_mesh(mesh)
    end_time = time.time()
    print_timing(f"Processing value {value}", end_time - start_time, should_print_timing)

def process_single_file(args):
    input_nrrd_path, max_distance, min_vertices, visualise, test, reconnection_mult, should_print_timing, should_fix_normals, replace = args
    start_time = time.time()
    
    # Create output directory
    output_obj_dir = os.path.join(os.path.dirname(input_nrrd_path), 'obj')
    os.makedirs(output_obj_dir, exist_ok=True)
    
    # Generate output path
    base_name = os.path.splitext(os.path.basename(input_nrrd_path))[0]
    output_obj_path = os.path.join(output_obj_dir, base_name)
    
    # Read NRRD file
    original_array, header = nrrd.read(input_nrrd_path)
    
    # Get space origin from header
    space_origin = header.get('space origin')
    if len(space_origin) == 3:
        space_origin = tuple(space_origin.astype(float))
    
    # Calculate average PCA normal
    avg_pca_normal = calculate_average_pca_normal(original_array)
    
    # Process each unique value
    array_values = np.unique(original_array)
    array_values = [v for v in array_values if v != 0]
    
    for value in array_values:
        if test:
            if value != 1 and value != 2:
                continue
        temp_output_obj_path = f'{output_obj_path}_{value}.obj'
        
        # Check if file exists and skip if not replacing
        if os.path.exists(temp_output_obj_path) and not replace:
            # print(f"Skipping existing file: {temp_output_obj_path}")
            continue
        
        array = original_array.copy()
        array[original_array != value] = 0
        mesh = array_to_thin_sheet_obj(array, temp_output_obj_path, max_distance=max_distance, min_vertices=min_vertices, space_origin=space_origin, reconnection_mult=reconnection_mult, avg_pca_normal=avg_pca_normal, should_print_timing=should_print_timing, should_fix_normals=should_fix_normals)
        if visualise:
            visualize_mesh(mesh)
    end_time = time.time()
    print_timing(f"Processing file {input_nrrd_path}", end_time - start_time, should_print_timing)

def midline_labels_to_obj(input_nrrd_path, output_obj_path, array_values=None, max_distance=1.5, min_vertices=1000, visualise=False, reconnection_mult=3):
    original_array, header = nrrd.read(input_nrrd_path)
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    if not array_values:
        array_values = np.unique(original_array)
    array_values = [v for v in array_values if v != 0]
    
    # Get space origin from header
    space_origin = header.get('space origin')
    if space_origin:
        space_origin = tuple(map(float, space_origin.strip('()').split(',')))
    
    # Calculate average PCA normal
    avg_pca_normal = calculate_average_pca_normal(original_array)
    
    # Prepare arguments for multiprocessing
    args_list = [(value, original_array, output_obj_path, max_distance, min_vertices, visualise, space_origin, reconnection_mult, avg_pca_normal) for value in array_values]
    
    # Use multiprocessing to process label values in parallel, 
    # but we are already parallel across cube files, seems to make no positive or negative difference
    with multiprocessing.Pool() as pool:
        pool.map(process_single_value, args_list)

    # for args in args_list:
    #     process_single_value(args)

def process_folder(input_folder, max_distance=1.5, min_vertices=500, visualise=False, test=False, reconnection_mult=3, should_print_timing=False, should_fix_normals=True, replace=False):
    # Find all *_mask_thinned.nrrd files in the input folder and its subfolders
    nrrd_files = glob.glob(os.path.join(input_folder, '**', '*_mask_thinned.nrrd'), recursive=True)
    nrrd_files = [f for f in nrrd_files if re.match(file_type_pattern, f)]
    
    if not nrrd_files:
        print(f"No matching *_mask_thinned.nrrd files found in {input_folder}")
        return
    
    # Prepare arguments for multiprocessing
    args_list = [(file, max_distance, min_vertices, visualise, test, reconnection_mult, should_print_timing, should_fix_normals, replace) for file in nrrd_files]
    
    # Use multiprocessing to process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, args) for args in args_list]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(args_list), desc="Processing files"):
            pass



# Main execution
if __name__ == "__main__":
    input_directory = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um'

    parser = argparse.ArgumentParser(description="Process NRRD files to OBJ meshes.")
    parser.add_argument("--vis", action="store_true", help="Visualize the meshes after creation for testing.")
    parser.add_argument("--test", action="store_true", help="Process only the first NRRD file found for testing.")
    parser.add_argument("--rcm", type=float, default=50, help="Reconnection multiplier for the surface.")
    parser.add_argument("--time", action="store_true", help="Print timing information")
    parser.add_argument("--nonormals", action="store_true", help="Disable normal fixing")
    parser.add_argument("--file_type", choices=['fb_avg', 'front', 'back', 'graph', 'dist_map', 'all'], 
                        default='all', help="Specify the type of NRRD file to process")
    parser.add_argument("--replace", action="store_true", help="Replace existing OBJ files")
    args = parser.parse_args()
    
    should_print_timing = args.time
    should_fix_normals = not args.nonormals
    
    stime = time.time()
    reconnection_mult = args.rcm
    
    if args.test:
        #temp testing
        path = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/02000_02256_04816/02000_02256_04816_fb_avg_mask_thinned.nrrd'
        process_single_file((path, 1.5, 500, args.vis, False, reconnection_mult, should_print_timing, should_fix_normals, args.replace))
        # p1 = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/01744_02000_04560/01744_02000_04560_graph_mask_thinned.nrrd'
        # process_single_file((p1, 1.5, 500, args.vis, False, reconnection_mult, should_print_timing, should_fix_normals))
        # p2 = '/Users/jamesdarby/Desktop/manually_labelled_cubes/public_s1-8um/01744_02000_04304/01744_02000_04304_dist_map_mask_thinned.nrrd'
        # process_single_file((p2, 1.5, 500, args.vis, False, reconnection_mult, should_print_timing, should_fix_normals))
    else:
        file_type_pattern = {
            'fb_avg': r'.*fb_avg_mask_thinned\.nrrd$',
            'front': r'.*front_mask_thinned\.nrrd$',
            'back': r'.*back_mask_thinned\.nrrd$',
            'graph': r'.*graph_mask_thinned\.nrrd$',
            'dist_map': r'.*dist_map_mask_thinned\.nrrd$',
            'all': r'.*_mask_thinned\.nrrd$'
        }[args.file_type]

        process_folder(input_directory, max_distance=1.5, min_vertices=500, visualise=args.vis, reconnection_mult=reconnection_mult, should_print_timing=should_print_timing, should_fix_normals=should_fix_normals, replace=args.replace)
    
    overall_end_time = time.time()
    print_timing("Total execution time", overall_end_time - stime, should_print_timing)