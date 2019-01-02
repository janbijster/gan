import PIL.Image
import os
import numpy as np
import matplotlib.pyplot as plt

# params
data_dir = 'data'
output_filename = 'data_bw.npy'
scale_images = False
output_size = (16, 16)
plot_images = False

# script
files = os.listdir(data_dir)

data = []
for file in files:
    if file[-4:] != '.jpg' and file[-4:] != '.png' and file[-4:] != '.gif':
        continue
    file_path = '{}/{}'.format(data_dir, file)
    print('Loading {}...'.format(file))
    try:
        im = PIL.Image.open(file_path).convert('L')
    except Exception:
        print('Error loading image, skipped.')
    if scale_images:
        im = im.resize(output_size, PIL.Image.BICUBIC)
    image = np.clip(np.array(im).astype('f') / 256., 0., 1.)
    if plot_images: 
        plt.matshow(image)
        plt.show()
    data.append(image)

    
train_matrix = np.array(data)
print(train_matrix.shape)
np.save(output_filename, train_matrix)