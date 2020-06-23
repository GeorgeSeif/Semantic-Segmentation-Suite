#https://github.com/yangsiyu007/SpaceNetExploration/blob/master/pipeline/split_train_val_test.py

import os
import random
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import cv2


from mask_utils import create_building_mask

from datetime import datetime


ds_path = Path("SpaceNet/")
raw_path = ds_path / 'raw_data'


image_dir_path = raw_path / 'PS-RGB'
label_dir_path = raw_path / 'geojson_buildings'

image_paths = list(image_dir_path.rglob('*.tif'))
label_paths = []


for image_path in image_paths:
    identifier = image_path.stem.split('PS-RGB_')[1] #e.g. "img28"
    label_path = label_dir_path / 'SN2_buildings_train_AOI_2_Vegas_geojson_buildings_{}.geojson'.format(identifier)
    label_paths.append(label_path)

# check if all corresponding geojson files exist
print('Checking all required geojson files exist')
for label_path in label_paths:
    if not (label_path).exists():
        print('{} does not exist'.format(label_path.name))
    
print('There are {} image files, {} geojson files'.format(len(image_paths), len(label_paths)))

images_labels = list(zip(image_paths, label_paths))
print('First pair before shuffle: {}'.format(images_labels[0]))
random.shuffle(images_labels) # in-place
print('First pair after shuffle: {}'.format(images_labels[0]))

train_len = int(len(images_labels) * 0.7)
val_len = int(len(images_labels) * 0.15)

splits = {}
splits['train'] = images_labels[:train_len]
splits['val'] = images_labels[train_len:train_len + val_len]
splits['test'] = images_labels[train_len + val_len:]

print('Resulting in {} train examples, {} val examples, {} test examples'.format(len(splits['train']), len(splits['val']), len(splits['test'])))


#create dirs

train_path = ds_path / 'train'
val_path = ds_path / 'val'
test_path = ds_path / 'test'

outputs = {}
outputs['train_label'] = ds_path / 'train_labels'
outputs['train_image'] = ds_path / 'train'
outputs['val_label'] = ds_path / 'val_labels'
outputs['val_image'] = ds_path / 'val'
outputs['test_label'] = ds_path / 'test_labels'
outputs['test_image'] = ds_path / 'test'

for name, output_dir in outputs.items():
	os.makedirs(output_dir, exist_ok=True)


start = datetime.utcnow()

for split_name in ['train', 'val', 'test']:
    print('Copying to {} output dir'.format(split_name))
    for image_path, label_path in splits[split_name]:
        # copy to correct split file
        
        #hutil.copy(label_path, Path(outputs['{}_label'.format(split_name)]) / label_path.name )
        #print("image_path", image_path)
        #print(outputs['{}_image'.format(split_name)])
        #print(image_path.name)
        

        # input_image = plt.imread(image_path)
        #input_image = cv2.imread('C:/Users/jschaffer/Documents/GitHub/Semantic-Segmentation-Suite/SpaceNet/raw_data/PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img4580.tif',1)

        #print(input_image)


        #plt.imshow(input_image)

        #vectorSrc = label_path
        mask_path = Path(outputs['{}_label'.format(split_name)]) / (label_path.stem.split('geojson_buildings_')[1]  + '_mask.tif')
        

        #print(mask_path)

        #print(str(image_path))

        #pixel_coords, latlon_coords = geojson_to_pixel_arr(str(image_path), str(label_path), pixel_ints=True,verbose=False)

        create_building_mask(str(image_path), str(label_path), npDistFileName=str(mask_path), perimeter_width=1)

        shutil.copy(image_path, Path(outputs['{}_image'.format(split_name)]) / image_path.name.split('PS-RGB_')[1] )

        


print("ETL completed in {} seconds".format(datetime.utcnow()-start))


#no perimeter 3:28
#permieter 4:11 / 4:36 / 4:12/ 4:18
