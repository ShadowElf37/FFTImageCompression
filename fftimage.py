"""
Compresses an image using FFT
"""

from PIL import Image
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
from multiprocessing import Pool
import multiprocessing as mp

import cft

pic= Image.open("9nsgatf9wga81.PNG").convert('RGB')
#pic= Image.open("mal2.jpg").convert('RGB')

print(np.array(pic.getdata()).shape)
pix = np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 3)

fig, (default, p1)=plt.subplots(2,4)#, p2, p3) = plt.subplots(2,4)

subplots = (*default, *p1)#, *p2, *p3)

for sp in subplots:
    sp.xaxis.tick_top()

default[0].imshow(pix)
default[1].imshow(pix[:,:,0], cmap='Reds_r')
default[2].imshow(pix[:,:,1], cmap='Greens_r')
default[3].imshow(pix[:,:,2], cmap='Blues_r')

"""
max_edge = int(np.sqrt(max_size))

b_red = f_red.copy()
b_red[max_edge:] = 0
b_red[:,max_edge:] = 0
b_red = np.abs(fft.ifft2(b_red))

b_green = f_green.copy()
b_green[max_edge:] = 0
b_green[:,max_edge:] = 0
b_green = np.abs(fft.ifft2(b_green))

b_blue = f_blue.copy()
b_blue[max_edge:] = 0
b_blue[:,max_edge:] = 0
b_blue = np.abs(fft.ifft2(b_blue))
"""

precision = np.csingle
#cft_pool = Pool(3)
res = 1/(2*np.pi)**2
(I1, I2,) = (res, res*pic.size[1]/pic.size[0],)

def compress(img, size, uv_cutoff=0, ir_cutoff=0, manual_excludes=()):
    """f_red = fft.fft2(img[:, :, 0]).astype(precision)
    f_green = fft.fft2(img[:, :, 1]).astype(precision)
    f_blue = fft.fft2(img[:, :, 2]).astype(precision)"""
    f_red = cft.Transformer(img[:, :, 0], (I1,I2)).cft(0,1).data.astype(precision)
    f_green = cft.Transformer(img[:, :, 1], (I1,I2)).cft(0,1).data.astype(precision)
    f_blue = cft.Transformer(img[:, :, 2], (I1,I2)).cft(0,1).data.astype(precision)


    #f_green = cft.Transformer(img[:, :, 1], (I1,I2)).cft(0,1).data.astype(precision)
    #f_blue = cft.Transformer(img[:, :, 2], (I1,I2)).cft(0,1).data.astype(precision)

    if uv_cutoff:
        f_red[uv_cutoff:, :] = 0
        f_red[:, uv_cutoff:] = 0
        f_green[uv_cutoff:, :] = 0
        f_green[:, uv_cutoff:] = 0
        f_blue[uv_cutoff:, :] = 0
        f_blue[:, uv_cutoff:] = 0
    if ir_cutoff or True:
        f_red[:ir_cutoff, :] = 0
        f_red[:, :ir_cutoff] = 0
        f_green[:ir_cutoff, :] = 0
        f_green[:, :ir_cutoff] = 0
        f_blue[:ir_cutoff, :] = 0
        f_blue[:, :ir_cutoff] = 0

    for exclude in manual_excludes:
        f_red[exclude[0]:exclude[1], :] = 0
        f_red[:, exclude[0]:exclude[1]] = 0
        f_blue[exclude[0]:exclude[1], :] = 0
        f_blue[:, exclude[0]:exclude[1]] = 0
        f_green[exclude[0]:exclude[1], :] = 0
        f_green[:, exclude[0]:exclude[1]] = 0

    """global p1
    p1[0].imshow((np.power(np.abs(np.stack((f_red, f_green, f_blue), axis=2))/np.median(np.abs(f_red)), 2)*255).astype(int))
    p1[1].imshow(np.abs(f_red)*255/np.median(np.abs(f_red)), vmin=0, vmax=255, cmap='Reds_r')
    p1[2].imshow(np.abs(f_green)*255/np.median(np.abs(f_red)), vmin=0, vmax=255, cmap='Greens_r')
    p1[3].imshow(np.abs(f_blue)*255/np.median(np.abs(f_red)), vmin=0, vmax=255, cmap='Blues_r')"""

    lower_bound = -np.partition(-np.concatenate((np.abs(f_red).flatten(), np.abs(f_green).flatten(), np.abs(f_blue).flatten())), size-1)[size-1]
    f_red[np.less(np.abs(f_red), lower_bound)] = 0
    f_green[np.less(np.abs(f_green), lower_bound)] = 0
    f_blue[np.less(np.abs(f_blue), lower_bound)] = 0



    r = list(zip(*list(np.nonzero(f_red))))
    r = np.array(list(zip(r, (f_red[i] for i in r))))
    g = list(zip(*list(np.nonzero(f_green))))
    g = np.array(list(zip(g, (f_green[i] for i in g))))
    b = list(zip(*list(np.nonzero(f_blue))))
    b = np.array(list(zip(b, (f_blue[i] for i in b))))
    #print(b)
    #print(r)

    print(f'Compression complete. '
          f'Original: {img.size} bytes. '
          f'Compressed: {getsizeof(r) + getsizeof(g) + getsizeof(b)} bytes. '
          f'Ratio: {round((getsizeof(r) + getsizeof(g) + getsizeof(b)) / img.size * 100, 3)}%')
    return f_red.shape, r,g,b

def decompress(shape, r, g, b):
    reds = np.zeros(shape, dtype=precision)
    blues = np.zeros(shape, dtype=precision)
    greens = np.zeros(shape, dtype=precision)

    for i in r:
        reds[i[0]] = i[1]
    for i in g:
        greens[i[0]] = i[1]
    for i in b:
        blues[i[0]] = i[1]

    """c_red = np.abs(fft.ifft2(reds))
    c_green = np.abs(fft.ifft2(greens))
    c_blue = np.abs(fft.ifft2(blues))"""

    c_red = np.abs(cft.Transformer(reds, (I1,I2)).icft(0,1).data)
    c_green = np.abs(cft.Transformer(greens, (I1,I2)).icft(0,1).data)
    c_blue = np.abs(cft.Transformer(blues, (I1,I2)).icft(0,1).data)

    c_pix = compose(c_red, c_green, c_blue)

    return c_pix, c_red, c_green, c_blue

def compose(r, g, b):
    return np.stack((r, g, b), axis=2).astype(int)

def weighter(n):
    a = 1-np.abs(np.linspace(-1, 1, n))
    return a/sum(a)
def rolling_filter(arr, n=5, axes=(0,1)):
    return np.apply_over_axes(
        lambda m,ax: np.apply_along_axis(lambda l: np.convolve(l, weighter(n), mode='full'), axis=ax, arr=m),
        arr, axes=axes)

for i,p in enumerate((p1,)):
    c_pix, c_red, c_green, c_blue = decompress(*compress(pix, 10 ** (i+5), uv_cutoff=0, ir_cutoff=0, manual_excludes=()))
    p[0].imshow(c_pix)
    p[1].imshow(c_red, cmap='Reds_r')
    p[2].imshow(c_green, cmap='Greens_r')
    p[3].imshow(c_blue, cmap='Blues_r')

"""
DENOISING_BIN = 5
denoised_r = rolling_filter(c_red, n=DENOISING_BIN)
denoised_g = rolling_filter(c_green, n=DENOISING_BIN)
denoised_b = rolling_filter(c_blue, n=DENOISING_BIN)
denoised_p = compose(denoised_r, denoised_g, denoised_b)

p2[0].imshow(denoised_p)
p2[1].imshow(denoised_r, cmap='Reds_r')
p2[2].imshow(denoised_g, cmap='Greens_r')
p2[3].imshow(denoised_b, cmap='Blues_r')

DENOISING_BIN = 20
denoised_r = rolling_filter(c_red, n=DENOISING_BIN)
denoised_g = rolling_filter(c_green, n=DENOISING_BIN)
denoised_b = rolling_filter(c_blue, n=DENOISING_BIN)
denoised_p = compose(denoised_r, denoised_g, denoised_b)

p3[0].imshow(denoised_p)
p3[1].imshow(denoised_r, cmap='Reds_r')
p3[2].imshow(denoised_g, cmap='Greens_r')
p3[3].imshow(denoised_b, cmap='Blues_r')"""



plt.show()