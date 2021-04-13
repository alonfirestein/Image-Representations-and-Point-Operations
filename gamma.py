from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB
import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    if rep == LOAD_GRAY_SCALE:
        image = cv2.imread(img_path, 0)/255
    elif rep == LOAD_RGB:
        image = cv2.imread(img_path)/255
    else:
        raise ValueError('Only RGB or GRAY_SCALE photo representations')

    def on_trackbar(GammaValue):
        gamma = GammaValue/100
        gamma_corrected_image = np.power(image, gamma)
        cv2.imshow('Gamma Correction', gamma_corrected_image)

    cv2.namedWindow('Gamma Correction')
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, 'Gamma Correction', 100, 200, on_trackbar)
    gamma_slider_pos = 100
    # Starting the image at a gamma position that looks normal from the start
    on_trackbar(gamma_slider_pos)
    # Keeping the window open until pressing any key
    cv2.waitKey()


def main():
    gammaDisplay('beach.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
