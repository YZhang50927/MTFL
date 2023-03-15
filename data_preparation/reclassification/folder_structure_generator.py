# Generate folder structure of VAD3
import os

def get_folder_structure(folder_path, indent=0):
    folder_structure = ''
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_structure += '{}{}/\n'.format('  ' * indent, item)
            folder_structure += get_folder_structure(item_path, indent + 1)
        else:
            folder_structure += '{}{}\n'.format('  ' * indent, item)
    return folder_structure

folder_path = '/media/mount_loc/yiling/VAD3'

with open('folder_structure.txt', 'w') as f:
    f.write(get_folder_structure(folder_path))
