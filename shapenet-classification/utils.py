import numpy as np

def read_pointnet_colors(seg_labels):
    ''' map segementation labels to colors '''
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors


CATEGORIES = {
    'Airplane': 0,
    'Bag': 1,
    'Car': 2,
    'Cap': 3,
    'Chair': 4,
    'Earphone':5,
    'Guitar': 6,
    'Knife': 7,
    'Lamp': 8,
    'Laptop': 9,
    'Motorbike': 10,
    'Mug': 11,
    'Pistol': 12,
    'Rocket': 13,
    'Skateboard': 14,
    'Table': 15,
}