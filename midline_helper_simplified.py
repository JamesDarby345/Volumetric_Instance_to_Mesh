import numpy as np
from scipy import ndimage
import graph_tool.all as gt
from sklearn.decomposition import PCA
import numba

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


def dist_map_max(distance_map):
    """
    For each x,y position on the z-axis, find the maximum value and set it to 1,
    while setting the rest to 0. If the maximum value is 0, that position is not included.

    Parameters:
    distance_map (numpy.ndarray): 3D distance map

    Returns:
    numpy.ndarray: Resulting 3D array with maximum values set to 1 and others to 0
    """
    result = np.zeros_like(distance_map)
    
    # Find the index of the maximum value along the z-axis
    max_indices = np.argmax(distance_map, axis=2)
    
    # Create a 3D index array
    x, y = np.indices(max_indices.shape)
    
    # Get the maximum values
    max_values = distance_map[x, y, max_indices]
    
    # Create a mask for positive maximum values
    non_zero_mask = max_values > 0
    
    # Set the maximum value positions to 1, only where the maximum value is non-zero
    result[x[non_zero_mask], y[non_zero_mask], max_indices[non_zero_mask]] = 1
    
    return result


def filter_and_reassign_labels(label_data, cc_min_size):
    """
    Filter out small disconnected components within each label and reassign
    remaining labels starting from 1 and incrementing by 1.

    Parameters:
    label_data (numpy.ndarray): Input label data
    cc_min_size (int): Minimum size for a connected component to be kept

    Returns:
    numpy.ndarray: Filtered and reassigned label data
    """
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    new_label_data = np.zeros_like(label_data)
    new_label = 1

    for label in unique_labels:
        label_mask = label_data == label
        labeled_components, _ = ndimage.label(label_mask)
        
        valid_component_mask = np.zeros_like(label_mask, dtype=bool)
        
        for component in range(1, labeled_components.max() + 1):
            component_mask = labeled_components == component
            if np.sum(component_mask) >= cc_min_size:
                valid_component_mask |= component_mask

        if np.any(valid_component_mask):
            new_label_data[valid_component_mask] = new_label
            new_label += 1

    return new_label_data

@numba.jit(nopython=True)
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

def generate_volume_roi(input_array, erode_dilate_iters=30):
    """
    Generates a Region of Interest (ROI) for a 3D volume by dilating the non-zero structure,
    filling morphological tunnels, and creating a mask that closely covers the volume.

    Args:
    input_array (numpy.ndarray): 3D input array
    erode_dilate_iters (int): Radius for dilation operation
    hole_size (int): Maximum size of holes/tunnels to fill

    Returns:
    numpy.ndarray: Binary mask representing the ROI
    """
    # Ensure the input is a 3D numpy array
    if input_array.ndim != 3:
        raise ValueError("Input must be a 3D numpy array")

    # Create a binary mask of non-zero elements
    binary_mask = (input_array > 0).astype(np.uint8)
    padded_structure = np.pad(binary_mask, pad_width=erode_dilate_iters, mode='constant', constant_values=0)

    # Dilate the binary mask
    dilated_mask = numba_dilation_3d_labels(padded_structure, erode_dilate_iters)
    # dilated_mask = ndimage.binary_dilation(padded_structure, 
    #                                        structure=ndimage.generate_binary_structure(3, 3),
    #                                        iterations=erode_dilate_iters)

    # Fill holes in the dilated mask
    filled_mask = ndimage.binary_fill_holes(dilated_mask)

    result = np.zeros_like(input_array, dtype=np.uint8)
    
    # dilate_iters = erode_dilate_iters -1
    
    eroded_padded_structure = ndimage.binary_erosion(filled_mask, iterations=erode_dilate_iters)

    eroded_structure = eroded_padded_structure[
        erode_dilate_iters:-erode_dilate_iters,
        erode_dilate_iters:-erode_dilate_iters,
        erode_dilate_iters:-erode_dilate_iters
    ]
    if eroded_structure.shape != input_array.shape:
        eroded_structure = np.zeros_like(input_array)
    result[eroded_structure] = 1
    
    # Create the final ROI mask
    roi_mask = result.astype(np.int8)
    roi_mask[roi_mask == 0] = -1

    return roi_mask

def create_masked_directed_energy_graph_from_dist_map(mask_data, direction='left', large_weight=1e8):
    z, y, x = mask_data.shape  # Dimensions of the 3D mask array
    # print(z, y, x)
    g = gt.Graph(directed=True)
    weight_prop = g.new_edge_property("int")  # Edge property for weights

    # Create vertex properties for i, j, k positions
    x_prop = g.new_vertex_property("int")
    y_prop = g.new_vertex_property("int")
    z_prop = g.new_vertex_property("int")

    # Create a mapping from mask coordinates to vertex indices
    coord_to_vertex = {}

    # Add vertices only for the non-zero elements in the mask
    # stime = time.time()
    # Find indices of non -1 elements using numpy vectorization
    non_neg_indices = np.argwhere(mask_data != -1)

    # Add all vertices at once
    g.add_vertex(len(non_neg_indices))
    
    # Assign vertices to coordinates and set properties
    for idx, (i, j, k) in enumerate(non_neg_indices):
        v = g.vertex(idx)
        coord_to_vertex[(i, j, k)] = v
        x_prop[v] = k
        y_prop[v] = j
        z_prop[v] = i

    # Define neighbor offsets based on directionality
    directions = {
        'left': [(0, 0, 1)],  # propagate right
        'right': [(0, 0, -1)],  # propagate left
        'top': [(0, 1, 0)],  # propagate downwards
        'bottom': [(0, -1, 0)],  # propagate upwards
        'front': [(1, 0, 0)],  # propagate back
        'back': [(-1, 0, 0)]  # propagate front
    }

    neighbors = directions[direction]
    
    edges = []
    weights = []

    # stime = time.time()

    for (i, j, k), current_vertex in coord_to_vertex.items():
        # Check each neighbor direction for valid connections
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if (ni, nj, nk) in coord_to_vertex:
                neighbor_vertex = coord_to_vertex[(ni, nj, nk)]
                # Determine edge weight from distance map, larger mask value means smaller weight
                # weight = int(1000/(mask_data[i, j, k]+1+1e-8))
                if mask_data[i, j, k] <= 0:
                    weight = 1e6
                else:
                    weight = int(1000/(mask_data[i, j, k]))
                # Add edge and assign weight
                edges.append((int(current_vertex), int(neighbor_vertex)))  # forward edge with energy value
                weights.append(weight)
                edges.append((int(neighbor_vertex), int(current_vertex)))  # backward edge with large energy value
                weights.append(int(large_weight))

        # Add each diagonal backwards neighbor inf edge, i.e., x-1, y-1 and x-1, y+1 for YX plane
        if (i, j-1, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j-1, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)
        if (i, j+1, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i, j+1, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)

        # Add each diagonal backwards neighbor inf edge for IK plane
        if (i-1, j, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i-1, j, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)
        if (i+1, j, k-1) in coord_to_vertex:
            neighbor_vertex = coord_to_vertex[(i+1, j, k-1)]
            edges.append((int(current_vertex), int(neighbor_vertex)))
            weights.append(int(large_weight) + 1)

    # Convert edges and weights to numpy arrays
    edges = np.array(edges, dtype=np.int32)
    weights = np.array(weights, dtype=np.int32)

    # Add edges to the graph using add_edge_list
    g.add_edge_list(edges)
    weight_prop.a = weights

    # print("Time taken to add edges to graph:", time.time()-stime)
    # stime = time.time()
    # Add source and sink nodes
    source = g.add_vertex()
    sink = g.add_vertex()

    # Helper function to get vertex indices for a face
    def get_face_vertices(face, coord_to_vertex, z, y, x):
        indices = []
        if face == 'left':
            for j in range(y):
                for k in range(x):
                    for i in range(z):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'right':
            for j in range(y):
                for k in range(x):
                    for i in range (z):
                        if (z-i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(z-i, j, k)])
                            break
        elif face == 'top':
            for i in range(z):
                for k in range(x):
                    for j in range(y):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'bottom':
            for i in range(z):
                for k in range(x):
                    for j in range(y):
                        if (i, y-j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, y-j, k)])
                            break
        elif face == 'front':
            for i in range(z):
                for j in range(y):
                    for k in range(x):
                        if (i, j, k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, k)])
                            break
        elif face == 'back':
            for i in range(z):
                for j in range(y):
                    for k in range(x):
                        if (i, j, x-k) in coord_to_vertex:
                            indices.append(coord_to_vertex[(i, j, x-k)])
                            break
        return indices

    # Connect source to 'front' face
    front_vertices = get_face_vertices('front', coord_to_vertex, z, y, x)
    for v in front_vertices:
        e = g.add_edge(source, v)
        weight_prop[e] = large_weight

    # Connect sink to 'back' face
    back_vertices = get_face_vertices('back', coord_to_vertex, z, y, x)
    for v in back_vertices:
        e = g.add_edge(v, sink)
        weight_prop[e] = large_weight

    g.edge_properties["weight"] = weight_prop
    # print("Time taken to add source and sink nodes:", time.time()-stime)
    return g, source, sink, weight_prop, x_prop, y_prop, z_prop

def find_boundary_vertices(edges, part):
    """
    Find vertices that cross the partition array border.
    
    Parameters:
    edges (np.ndarray): An array of edges, where each edge is represented by a tuple (source, target).
    part (np.ndarray): A partition array where part[i] is the partition of vertex i.
    
    Returns:
    set: A set of boundary vertices.
    """

    part = np.array(part.a)
    
    # Get the source and target vertices for each edge
    source_vertices = edges[:, 0]
    target_vertices = edges[:, 1]
    
    # Find edges that cross the partition border
    cross_partition = part[source_vertices] != part[target_vertices]
    # print("edges that cross the partition:", len(cross_partition))
    
    # Get the boundary vertices
    boundary_vertices = np.unique(np.concatenate((source_vertices[cross_partition], target_vertices[cross_partition])))
    boundary_vertices = boundary_vertices[part[boundary_vertices] == 0]
    # print("boundary vertices:", len(boundary_vertices))
    
    return set(boundary_vertices)

def boundary_vertices_to_array_masked(boundary_vertices, shape, face, x_pos, y_pos, z_pos):
    z_dim, y_dim, x_dim = shape
    boundary_array = np.zeros(shape, dtype=np.int8)

    #Compute the 3d coordinates from the x,y,z positions
    for vertex in boundary_vertices:
        # print(vertex)
        x = x_pos[vertex]
        y = y_pos[vertex]
        z = z_pos[vertex]

         # Check if indices are within the valid range
        if 0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim:
            boundary_array[z, y, x] = 1  # Mark the boundary vertex in the array
        else:
            print(f"Index out of bounds: z={z}, y={y}, x={x}")

    # Keep only the top-most value closest to the face in each column perpendicular to the face
    if face == 'x':
        for y in range(y_dim):
            for z in range(z_dim):
                row = boundary_array[z, y, :]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row


    elif face == 'y':
        for x in range(x_dim):
            for z in range(z_dim):
                row = boundary_array[z, :, x]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row

    elif face == 'z':
        for x in range(x_dim):
            for y in range(y_dim):
                row = boundary_array[:, y, x]
                if np.any(row == 1):
                    first_one_index = np.argmax(row == 1)
                    row[:first_one_index] = 0  # Set all values below to 0
                    row[first_one_index+1:] = 0  # Set all values above to 0
                else:
                    row[:] = 0  # No values found in this row

    return boundary_array

def connect_to_edge_3d(array, value, distance=1, use_x=True, use_y=True, use_z=True):
    """
    Connect values in a 3D array to the nearest edge using straight lines,
    if they are within the specified distance from the edge.
    When an axis is disabled, vertices closest to that axis are not connected.
    Optionally creates an outline using ray projection method.
    
    Args:
    array (numpy.ndarray): 3D input array
    value (int or float): The value to connect to the edge
    distance (int): Maximum distance from the edge to connect (default 1)
    use_x (bool): Whether to allow connections along the x-axis (default True)
    use_y (bool): Whether to allow connections along the y-axis (default True)
    use_z (bool): Whether to allow connections along the z-axis (default True)
    create_outline (bool): Whether to create the outline using ray projection (default False)
    
    Returns:
    numpy.ndarray: Modified 3D array with values connected to the edge and optional outline
    """
    # Create a copy of the input array
    result = np.copy(array)
    
    # Get the dimensions of the array
    depth, height, width = array.shape
    
    # Find coordinates of voxels with the specified value
    coords = np.argwhere(array == value)
    
    for z, y, x in coords:
        # Check if the voxel is within the specified distance from any edge
        if (z < distance or z >= depth - distance or
            y < distance or y >= height - distance or
            x < distance or x >= width - distance):
            
            # Determine the nearest edge for each dimension
            nearest_z = min(z, depth - 1 - z) if use_z else float('inf')
            nearest_y = min(y, height - 1 - y) if use_y else float('inf')
            nearest_x = min(x, width - 1 - x) if use_x else float('inf')
            
            # Find the dimension with the minimum distance to edge
            min_dist = min(nearest_z, nearest_y, nearest_x)
            
            # Only connect if the nearest edge is on an enabled axis
            if min_dist != float('inf'):
                if min_dist == nearest_z:
                    # Connect to the nearest z-edge
                    z_edge = 0 if z < depth // 2 else depth - 1
                    result[min(z, z_edge):max(z, z_edge)+1, y, x] = value
                elif min_dist == nearest_y:
                    # Connect to the nearest y-edge
                    y_edge = 0 if y < height // 2 else height - 1
                    result[z, min(y, y_edge):max(y, y_edge)+1, x] = value
                elif min_dist == nearest_x:
                    # Connect to the nearest x-edge
                    x_edge = 0 if x < width // 2 else width - 1
                    result[z, y, min(x, x_edge):max(x, x_edge)+1] = value
    
    return result
    
def calculate_orientation(array_3d, label_value):
    indices = np.where(array_3d == label_value)
    coordinates = np.array(indices).T
    pca = PCA(n_components=3)
    pca.fit(coordinates)
    perpendicular_direction = pca.components_[2]
    if np.sum(perpendicular_direction) < 0:
        perpendicular_direction = -perpendicular_direction
    return perpendicular_direction

def rotate_to_z_axis(array_3d, direction_vector):
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    axis_vectors = np.eye(3)
    dot_products = np.abs(np.dot(axis_vectors, direction_vector))
    closest_axis = np.argmax(dot_products)
    
    if closest_axis == 2:  # Already closest to z-axis
        return array_3d, (None, 0)
    elif closest_axis == 1:  # y-axis
        plane, k = (1, 2), 1  # yz-plane, 90 degree rotation
    else:  # x-axis
        plane, k = (0, 2), 1  # xz-plane, 90 degree rotation
    
    return np.rot90(array_3d, k=k, axes=plane), (plane, k)

def unapply_rotation(array_3d, rotation_info):
    plane, k = rotation_info
    if plane is None:
        return array_3d
    return np.rot90(array_3d, k=-k, axes=plane)

def get_slicer_colormap():
    return {
        0: [0, 0, 0, 0],
        1: [128, 174, 128, 255],
        2: [241, 214, 145, 255],
        3: [177, 122, 101, 255],
        4: [111, 184, 210, 255],
        5: [216, 101, 79, 255],
        6: [221, 170, 101, 255],
        7: [144, 238, 144, 255],
        8: [255, 181, 158, 255],
        9: [220, 245, 20, 255],
        10: [78, 63, 0, 255],
        11: [255, 250, 220, 255],
        12: [230, 220, 70, 255],
        13: [200, 200, 235, 255],
        14: [82, 174, 128, 255],
        15: [244, 214, 49, 255],
        16: [0, 151, 206, 255],
        17: [185, 232, 61, 255],
        18: [183, 156, 220, 255],
        19: [183, 214, 211, 255],
        20: [152, 189, 207, 255],
        21: [10, 255, 170, 255],
        22: [178, 212, 242, 255],
        23: [68, 172, 100, 255],
        24: [111, 197, 131, 255],
        25: [85, 188, 255, 255],
        26: [0, 145, 30, 255],
        27: [214, 230, 130, 255],
        28: [0, 147, 202, 255],
        29: [218, 255, 255, 255],
        30: [170, 250, 250, 255],
        31: [140, 224, 228, 255],
        32: [188, 65, 28, 255],
        33: [216, 191, 216, 255],
        34: [145, 60, 66, 255],
        35: [150, 98, 83, 255],
        36: [200, 200, 215, 255],
        37: [68, 131, 98, 255],
        38: [83, 146, 164, 255],
        39: [162, 115, 105, 255],
        40: [141, 93, 137, 255],
        41: [182, 166, 110, 255],
        42: [188, 135, 166, 255],
        43: [154, 150, 201, 255],
        44: [177, 140, 190, 255],
        45: [30, 111, 85, 255],
        46: [210, 157, 166, 255],
        47: [48, 129, 126, 255],
        48: [98, 153, 112, 255],
        49: [69, 110, 53, 255],
        50: [166, 113, 137, 255],
        51: [122, 101, 38, 255],
        52: [253, 135, 192, 255],
        53: [145, 92, 109, 255],
        54: [46, 101, 131, 255],
        55: [250, 250, 225, 255],
        56: [127, 150, 88, 255],
        57: [159, 116, 163, 255],
        58: [125, 102, 154, 255],
        59: [106, 174, 155, 255],
        60: [154, 146, 83, 255],
        61: [126, 126, 55, 255],
        62: [201, 160, 133, 255],
        63: [78, 152, 141, 255],
        64: [174, 140, 103, 255],
        65: [139, 126, 177, 255],
        66: [148, 120, 72, 255],
        67: [186, 135, 135, 255],
        68: [99, 106, 24, 255],
        69: [156, 171, 108, 255],
        70: [64, 123, 147, 255],
        71: [138, 95, 74, 255],
        72: [97, 113, 158, 255],
        73: [126, 161, 197, 255],
        74: [194, 195, 164, 255],
        75: [88, 106, 215, 255],
        76: [244, 214, 49, 255],
        77: [200, 200, 215, 255],
        78: [241, 172, 151, 255],
        79: [57, 157, 110, 255],
        80: [60, 143, 83, 255],
        81: [92, 162, 109, 255],
        82: [255, 244, 209, 255],
        83: [201, 121, 77, 255],
        84: [70, 163, 117, 255],
        85: [188, 91, 95, 255],
        86: [166, 84, 94, 255],
        87: [182, 105, 107, 255],
        88: [229, 147, 118, 255],
        89: [174, 122, 90, 255],
        90: [201, 112, 73, 255],
        91: [194, 142, 0, 255],
        92: [241, 213, 144, 255],
        93: [203, 179, 77, 255],
        94: [229, 204, 109, 255],
        95: [255, 243, 152, 255],
        96: [209, 185, 85, 255],
        97: [248, 223, 131, 255],
        98: [255, 230, 138, 255],
        99: [196, 172, 68, 255],
        100: [255, 255, 167, 255],
        101: [255, 250, 160, 255],
        102: [255, 237, 145, 255],
        103: [242, 217, 123, 255],
        104: [222, 198, 101, 255],
        105: [213, 124, 109, 255],
        106: [184, 105, 108, 255],
        107: [150, 208, 243, 255],
        108: [62, 162, 114, 255],
        109: [242, 206, 142, 255],
        110: [250, 210, 139, 255],
        111: [255, 255, 207, 255],
        112: [182, 228, 255, 255],
        113: [175, 216, 244, 255],
        114: [197, 165, 145, 255],
        115: [172, 138, 115, 255],
        116: [202, 164, 140, 255],
        117: [224, 186, 162, 255],
        118: [255, 245, 217, 255],
        119: [206, 110, 84, 255],
        120: [210, 115, 89, 255],
        121: [203, 108, 81, 255],
        122: [233, 138, 112, 255],
        123: [195, 100, 73, 255],
        124: [181, 85, 57, 255],
        125: [152, 55, 13, 255],
        126: [159, 63, 27, 255],
        127: [166, 70, 38, 255],
        128: [218, 123, 97, 255],
        129: [225, 130, 104, 255],
        130: [224, 97, 76, 255],
        131: [184, 122, 154, 255],
        132: [211, 171, 143, 255],
        133: [47, 150, 103, 255],
        134: [173, 121, 88, 255],
        135: [188, 95, 76, 255],
        136: [255, 239, 172, 255],
        137: [226, 202, 134, 255],
        138: [253, 232, 158, 255],
        139: [244, 217, 154, 255],
        140: [205, 179, 108, 255],
        141: [186, 124, 161, 255],
        142: [255, 255, 220, 255],
        143: [234, 234, 194, 255],
        144: [204, 142, 178, 255],
        145: [180, 119, 153, 255],
        146: [216, 132, 105, 255],
        147: [255, 253, 229, 255],
        148: [205, 167, 142, 255],
        149: [204, 168, 143, 255],
        150: [255, 224, 199, 255],
        151: [0, 145, 30, 255],
        152: [139, 150, 98, 255],
        153: [249, 180, 111, 255],
        154: [157, 108, 162, 255],
        155: [203, 136, 116, 255],
        156: [185, 102, 83, 255],
        157: [247, 182, 164, 255],
        158: [222, 154, 132, 255],
        159: [124, 186, 223, 255],
        160: [249, 186, 150, 255],
        161: [244, 170, 147, 255],
        162: [192, 104, 88, 255],
        163: [255, 190, 165, 255],
        164: [227, 153, 130, 255],
        165: [213, 141, 113, 255],
        166: [193, 123, 103, 255],
        167: [216, 146, 127, 255],
        168: [230, 158, 140, 255],
        169: [245, 172, 147, 255],
        170: [250, 250, 210, 255],
        171: [177, 124, 92, 255],
        172: [171, 85, 68, 255],
        173: [217, 198, 131, 255],
        174: [212, 188, 102, 255],
        175: [185, 135, 134, 255],
        176: [198, 175, 125, 255],
        177: [194, 98, 79, 255],
        178: [194, 98, 79, 255],
        179: [255, 226, 77, 255],
        180: [224, 194, 0, 255],
        181: [0, 147, 202, 255],
        182: [240, 255, 30, 255],
        183: [185, 232, 61, 255],
    }