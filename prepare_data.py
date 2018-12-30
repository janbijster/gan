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
        im = PIL.Image.open(file_path).convert('L') # grayscale for now
    except Exception:
        print('Error loading image, skipped.')
    im_scaled = im.resize(output_size, PIL.Image.BICUBIC)
    image = np.array(im_scaled)
    data.append(image)

    
train_matrix = np.array(data)
print(train_matrix.shape)
np.save(output_filename, train_matrix)