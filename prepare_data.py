import PIL.Image
import os
import numpy as np

# params
data_dir = 'data'
output_size = (28, 28)
output_filename = 'data.npy'

# script
files = os.listdir(data_dir)

data = []
for file in files:
    if file[-4:] != '.jpg' and file[-4:] != '.png':
        continue
    file_path = '{}/{}'.format(data_dir, file)
    print('Loading {}...'.format(file))
    try:
        im = PIL.Image.open(file_path)
        im_scaled = im.resize(output_size, PIL.Image.BICUBIC)
    except Exception:
        print('Error loading image, skipped.')
    image = np.array(im_scaled)
    if image.shape == (28, 28, 3):
        data.append(image)
    else:
        print('Image shape not right, skipping...')

    
train_matrix = np.array(data)
print(train_matrix.shape)
np.save(output_filename, train_matrix)