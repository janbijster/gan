import PIL.Image
import os
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

# params
data_dir = 'data'
output_size = (16, 16)
scale_images = False
output_filename = 'data.npy'
convert_hsv = False
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
        im = PIL.Image.open(file_path).convert('RGB')
    except Exception:
        print('Error loading image, skipped.')
    if scale_images:
        im = im.resize(output_size, PIL.Image.BICUBIC)
    image = np.array(im)
    image = np.clip(image.astype('f') / 256., 0., 1.)
    if plot_images: 
        plt.imshow(im)
        plt.show()
    if convert_hsv:
        image = matplotlib.colors.rgb_to_hsv(image)
    #print(image.shape)
    #print((output_size[0], output_size[1], 3))
    if image.shape == (output_size[0], output_size[1], 3):
        data.append(image)
    else:
        print('Image shape not right, skipping...')

    
train_matrix = np.array(data)
print(train_matrix.shape)
np.save(output_filename, train_matrix)