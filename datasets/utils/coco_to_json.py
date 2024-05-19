# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 21:15:30 2023

@author: evrim
"""

import json
import os

def coco_to_labelme(coco_file, output_dir):
    # Load COCO JSON
    with open(coco_file, 'r') as file:
        coco = json.load(file)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary for mapping category ids to names
    categories = {category['id']: category['name'] for category in coco['categories']}

    # Process each image
    for cnt, image in enumerate(coco['images']):
        labelme_data = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [],
            'imagePath': image['file_name'],
            'imageData': None,
            'imageHeight': image['height'],
            'imageWidth': image['width']
        }
        
        if os.path.isfile(os.path.join(output_dir, f"{image['file_name'].split('.')[0]}.json")):
            continue

        # Find annotations for this image
        annotations = [anno for anno in coco['annotations'] if anno['image_id'] == image['id']]
        
        for anno in annotations:
            # Check if 'segmentation' is non-empty and is a list
            if 'segmentation' in anno and anno['segmentation'] and isinstance(anno['segmentation'], list):
                shape = {
                    'label': categories[anno['category_id']],
                    'points': anno['segmentation'][0],
                    'group_id': None,
                    'shape_type': 'polygon',
                    'flags': {}
                }
                labelme_data['shapes'].append(shape)

        # Write LabelMe JSON
        labelme_file = os.path.join(output_dir, f"{image['file_name'].split('.')[0]}.json")
        with open(labelme_file, 'w') as file:
            json.dump(labelme_data, file, indent=4)
            
        print(f"{cnt}/{len(coco['images'])}")

# Example usage
coco_file = os.path.join("coco2017", "annotations", "instances_train2017.json")
output_dir = os.path.join("coco2017", "annotations", "labelme_annotations")
coco_to_labelme(coco_file, output_dir)
