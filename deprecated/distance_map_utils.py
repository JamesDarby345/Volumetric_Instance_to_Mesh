import numpy as np
from scipy import ndimage
import concurrent.futures

def process_label(labeled_array, label):
    mask = (labeled_array == label)
    distance_map = ndimage.distance_transform_edt(mask)
    distance_map[~mask] = 0
    return distance_map

def process_label_with_roi(labeled_array, label, roi_array):
    mask = (labeled_array == label)
    pos_distance_map = ndimage.distance_transform_edt(mask)
    neg_distance_map = ndimage.distance_transform_edt(~mask)
    distance_map = pos_distance_map - neg_distance_map
    roi_mask = (roi_array != 0)
    distance_map[~roi_mask] = 0
    return distance_map

def create_label_distance_map(labeled_array, max_workers=None):
    labels = np.unique(labeled_array)
    labels = labels[labels != 0]
    output = np.zeros_like(labeled_array, dtype=float)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {executor.submit(process_label, labeled_array, label): label for label in labels}
        
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                distance_map = future.result()
                output += distance_map
            except Exception as exc:
                print(f'Label {label} generated an exception: {exc}')
    
    return output.astype(int)

def create_label_distance_map_with_roi(labeled_array, roi_array, max_workers=None):
    assert labeled_array.shape == roi_array.shape, "labeled_array and roi_array must have the same shape"
    
    labels = np.unique(labeled_array)
    labels = labels[labels != 0]
    output = np.zeros_like(labeled_array, dtype=float)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {executor.submit(process_label_with_roi, labeled_array, label, roi_array): label for label in labels}
        
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                distance_map = future.result()
                output += distance_map
            except Exception as exc:
                print(f'Label {label} generated an exception: {exc}')
    
    return output

def create_single_label_distance_map(labeled_array, label):
    mask = (labeled_array == label)
    distance_map = ndimage.distance_transform_edt(mask)
    distance_map[~mask] = 0
    return distance_map

def prepare_distance_map(distance_map, roi_mask, value_to_add=0):
    distance_map[distance_map > 5] = 5
    mask = (distance_map > 0)
    distance_map[mask] += value_to_add

    distance_map[roi_mask == -1] = -1
    distance_map += abs(distance_map.min()) + 1
    distance_map[roi_mask == -1] = -1
    return distance_map