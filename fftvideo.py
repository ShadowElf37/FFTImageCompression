"""
Compresses a video using FFT
"""

from PIL import Image
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from video import reader
import matplotlib.animation as animation
from sys import getsizeof

#r = reader('test.mov')
r = reader('pikachu.mov')
frames = np.array(list(r))[:]
#print(frames[123,380,600])
frames[:,:,:,[2,1,0]] = frames[:,:,:,[0,1,2]]
#print('BREAK')
#print(frames[123,380,600])
#exit()

#del frames
#pic= Image.open("mal2.jpg").convert('RGB')

fig, (default, p1) = plt.subplots(2,4)

subplots = (*default, *p1)

for sp in subplots:
    sp.xaxis.tick_top()

#frame = next(r)

precision = np.csingle

def compress(img, size, uv_cutoff=0, ir_cutoff=0, manual_excludes=()):
    print('Beginning compression.')
    print('Transforming red...')
    f_red = fft.fftn(img[:, :, :, 0]).astype(precision)
    print('Transforming green...')
    f_green = fft.fftn(img[:, :, :, 1]).astype(precision)
    print('Transforming blue...')
    f_blue = fft.fftn(img[:, :, :, 2]).astype(precision)

    print('Doing exclusions...')
    if uv_cutoff:
        f_red[:, uv_cutoff:, :] = 0
        f_red[:, :, uv_cutoff:] = 0
        f_green[:, uv_cutoff:, :] = 0
        f_green[:, :, uv_cutoff:] = 0
        f_blue[:, uv_cutoff:, :] = 0
        f_blue[:, :, uv_cutoff:] = 0
    if ir_cutoff or True:
        f_red[:, :ir_cutoff, :] = 0
        f_red[:, :, :ir_cutoff] = 0
        f_green[:, :ir_cutoff, :] = 0
        f_green[:, :, :ir_cutoff] = 0
        f_blue[:, :ir_cutoff, :] = 0
        f_blue[:, :, :ir_cutoff] = 0

    for exclude in manual_excludes:
        f_red[:, exclude[0]:exclude[1], :] = 0
        f_red[:, :, exclude[0]:exclude[1]] = 0
        f_blue[:, exclude[0]:exclude[1], :] = 0
        f_blue[:, :, exclude[0]:exclude[1]] = 0
        f_green[:, exclude[0]:exclude[1], :] = 0
        f_green[:, :, exclude[0]:exclude[1]] = 0

    print('Finding strongest signals...')
    lower_bound = - \
    np.partition(-np.concatenate((np.abs(f_red).flatten(), np.abs(f_green).flatten(), np.abs(f_blue).flatten())),
                 size - 1)[size - 1]
    f_red[np.less(np.abs(f_red), lower_bound)] = 0
    f_green[np.less(np.abs(f_green), lower_bound)] = 0
    f_blue[np.less(np.abs(f_blue), lower_bound)] = 0

    print('Collecting coordinates...')
    r = list(zip(*list(np.nonzero(f_red))))
    r = np.array(list(zip(r, [f_red[i] for i in r])))
    g = list(zip(*list(np.nonzero(f_green))))
    g = np.array(list(zip(g, [f_green[i] for i in g])))
    b = list(zip(*list(np.nonzero(f_blue))))
    b = np.array(list(zip(b, [f_blue[i] for i in b])))
    #print(b)
    #print(r)
    print(f'Compression complete. '
          f'Original: {img.size} bytes. '
          f'Compressed: {getsizeof(r)+getsizeof(g)+getsizeof(b)} bytes. '
          f'Ratio: {round((getsizeof(r)+getsizeof(g)+getsizeof(b))/img.size*100, 3)}%')

    return f_red.shape, r,g,b

def decompress(shape, r, g, b):
    print('Beginning decompression.')
    reds = np.zeros(shape, dtype=precision)
    blues = np.zeros(shape, dtype=precision)
    greens = np.zeros(shape, dtype=precision)

    print(shape)

    for i in r:
        reds[i[0]] = i[1]
    for i in g:
        greens[i[0]] = i[1]
    for i in b:
        blues[i[0]] = i[1]

    print('Transforming red...')
    c_red = np.abs(fft.ifftn(reds))
    print('Transforming green...')
    c_green = np.abs(fft.ifftn(greens))
    print('Transforming blue...')
    c_blue = np.abs(fft.ifftn(blues))

    print('Composing final image...')
    c_pix = np.stack((c_red, c_green, c_blue), axis=3).astype(int)

    print('Decompression complete.')
    return c_pix#, c_red, c_green, c_blue



d1 = default[0].imshow(frames[0])
d2 = default[1].imshow(frames[:,:,:,0][0], cmap='Reds_r')
d3 = default[2].imshow(frames[:,:,:,1][0], cmap='Greens_r')
d4 = default[3].imshow(frames[:,:,:,2][0], cmap='Blues_r')

defaults = [d1, d2, d3, d4]

c_frames = decompress(*compress(frames, 150000, uv_cutoff=0, ir_cutoff=0, manual_excludes=())) # , c_red, c_green, c_blue
img1 = p1[0].imshow(c_frames[0])
img2 = p1[1].imshow(c_frames[:,:,:,0][0], vmax=255, cmap='Reds_r')
img3 = p1[2].imshow(c_frames[:,:,:,1][0], vmax=255, cmap='Greens_r')
img4 = p1[3].imshow(c_frames[:,:,:,2][0], vmax=255, cmap='Blues_r')

compressed = [img1, img2, img3, img4]

images = defaults + compressed
all_frames = np.array([frames, c_frames])

def animate_func(i, img_index):
    #print('Animating', img_index)
    img = images[img_index]
    img.set_array(all_frames[int(img_index>3)][i,:,:,slice(None) if img_index % 4 == 0 else (img_index%4-1)%3])
    #print(i)
    return [img]


print('Compressed', getsizeof(frames), 'bytes to', getsizeof(c_frames))

anims = [animation.FuncAnimation(
                               fig,
                               (lambda j: (lambda i: animate_func(i, j)))(k),
                               frames = frames.shape[0],
                               interval = 50, # in ms
                               ) for k in range(0, 8)]

plt.show()