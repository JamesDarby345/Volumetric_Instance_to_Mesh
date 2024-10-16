import os
import shutil
import argparse

def copy_mask_files(input_folder, output_folder, suffix=None):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if suffix:
                if file.endswith(suffix):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output_folder, file)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path} -> {dst_path}")
            else:
                if file.endswith('_mask.nrrd') and 'relabeled' not in file and 'morph' not in file:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(output_folder, file)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path} -> {dst_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy specific mask files to an output folder.')
    parser.add_argument('--input_folder', type=str, default='./relabeled_cubes', help='Path to the input folder')
    parser.add_argument('--output_folder', type=str, default='./temp_move_cubes', help='Path to the output folder')
    parser.add_argument('--suffix', type=str, help='Exact suffix to match (e.g., "_mask.nrrd")')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    copy_mask_files(args.input_folder, args.output_folder, args.suffix)
