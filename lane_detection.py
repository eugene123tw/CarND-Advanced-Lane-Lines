import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tools import cal_undistort, unwrap, binarization, find_peaks


def main():


    CAL_fnames = glob.glob('camera_cal/*.jpg')
    test_fnames = glob.glob('test_images/*.jpg')
    mtx, dist = cal_undistort(CAL_fnames, 9, 6)

    for fname in test_fnames:
        test_img = cv2.imread(fname)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)

        binary = binarization(undistorted, s_thresh=(210, 240))

        unwrapped_undistorted, M = unwrap(binary)


        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(test_img)
        ax1.set_title('Original Image', fontsize=10)
        ax2.imshow(unwrapped_undistorted, cmap='gray')
        ax2.set_title('Binarized Unwrap Image', fontsize=10)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        find_peaks(unwrapped_undistorted)



if __name__ == '__main__':
    main()
