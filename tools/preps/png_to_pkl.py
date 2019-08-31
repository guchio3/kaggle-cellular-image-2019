import gc
import os
from functools import partial
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


def _png_to_npy(filename):
    npy_filename = filename.replace('.png', '.npy')
#    if not os.path.exists(npy_filename):
    img = cv2.imread(filename, 1)
    np.save(npy_filename, img)


def png_to_npy():
    filenames = glob('./mnt/inputs/train/*/*/*')
    filenames += glob('./mnt/inputs/test/*/*/*')
    print(len(filenames))
    with Pool(os.cpu_count()) as p:
        iter_func = partial(_png_to_npy)
        imap = p.imap_unordered(iter_func, filenames)
        _ = list(tqdm(imap, total=len(filenames)))
        p.close()
        p.join()
        gc.collect()


if __name__ == '__main__':
    png_to_npy()
