from typing import List
import cv2 as cv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from) ===> nice one
    :return: int
    """
    return 314984402


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE(1) or RGB(2)
    :return: The image object
    """
    # Loading an image and converting it according the the representation input
    img = cv.imread(filename)
    if img is not None:
        if representation == LOAD_GRAY_SCALE:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        elif representation == LOAD_RGB:
            # We weren't asked to convert a grayscale image to RGB so this will suffice
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:  # Any other value was entered as the second parameter
            raise ValueError("Please enter [1] for Grayscale, or [2] for RGB representation of the image.")
    else:
        raise Exception("Could not read the image! Please try again.")
    return img / 255.0
    # Decided to divide the result by 255 instead of using the cv2.normalize function


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE(1) or RGB(2)
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    if representation == LOAD_GRAY_SCALE:
        plt.gray()
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Making sure the shape of the image is height*width*3
    if len(imgRGB.shape) == 3:
        matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
        imgYIQ = imgRGB.dot(matrix.T)
        # To show the image: Uncomment the next two lines
        # plt.imshow(imgYIQ)
        # plt.show()
        return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Making sure the shape of the image is height*width*3
    # Formula for conversion taken from here: https://en.wikipedia.org/wiki/YIQ
    if len(imgYIQ.shape) == 3:
        matrix = np.array([[1.0, 0.956, 0.619],
                           [1.0, -0.272, -0.647],
                           [1.0, -1.106, 1.703]])
        imgRGB = imgYIQ.dot(matrix.T)
        # To show the image: Uncomment the next two lines
        # plt.imshow(imgRGB)
        # plt.show()
        return imgRGB


def histogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original Histogram
    :return (imgEq,histOrg,histEQ)
    where:
    • imOrig - is the input grayscale or RGB image to be equalized having values in the range [0, 1].
    • imEq - is the new image with the equalized histogram
    • histOrg - is the histogram of the original image
    • histEQ - is the histogram of the imgEq
    """
    img = imgOrig
    # If an RGB image is given the following equalization procedure should only operate on the Y channel of
    # the corresponding YIQ image and then convert back from YIQ to RGB.
    if len(imgOrig.shape) == 3:
        img = cv.normalize(img.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
        img = transformRGB2YIQ(img)
        imgEq = img[:, :, 0]
        imgEq = cv.normalize(imgEq, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    # Image is grayscale
    else:
        imgEq = (img * 255).astype('uint8')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Original CDF
    histOrg, bins = np.histogram(imgEq.flatten(), 256, [0, 256])
    cdf = histOrg.cumsum()  # Calculate cumulative histogram
    cdf_normalized = cdf * histOrg.max() / cdf.max()
    ax1.set_title('Original Image Histogram & CDF')
    ax1.plot(cdf_normalized, color='navy')
    ax1.hist(imgEq.flatten(), 256, [0, 256], color='red')
    ax1.legend(('CDF', 'Histogram'), loc='best')

    # Equalized Image with linear CDF
    cdf_m = ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = ma.filled(cdf_m, 0).astype('uint8')
    linear_Img = imgEq
    linear_Img = cdf[linear_Img]
    histEq, bins = np.histogram(linear_Img.flatten(), 256, [0, 255])
    cdf_linear = histEq.cumsum()
    cdf_normalized_lin = cdf_linear * histEq.max() / cdf_linear.max()
    ax2.set_title('Post-Equalization Image Histogram & CDF ')
    ax2.plot(cdf_normalized_lin, color='navy')
    ax2.hist(linear_Img.flatten(), 256, [0, 255], color='red')
    ax2.legend(('CDF', 'Histogram'), loc='best')

    plt.xlim([0, 256])
    plt.show()

    # Needing to Transform equalized image back to RGB
    if len(imgOrig.shape) == 3:
        imgEq = cv.normalize(imgEq.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
        img[:, :, 0] = imgEq
        imgEq = transformYIQ2RGB(img)
        imgEq = (imgEq * 255).astype('uint8')

    imgEq = cv.LUT(imgEq, cdf)
    return imgEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # If an RGB image is given, the following quantization procedure should only operate on the Y channel
    # of the corresponding YIQ image and then convert back from YIQ to RGB.
    # Therefore when we get an RGB image ,we take only the y channel here:
    imOrig *= 255
    if len(imOrig.shape) == 3:
        YIQ = transformRGB2YIQ(imOrig)
        img = YIQ[:, :, 0].copy()
    # Image is grayscale
    else:
        img = imOrig.copy()
    # Creating the Z vector with nQuant+1 elements, First element is 0, and last element is 255 respectively
    borders = np.array([0, 255])
    for i in range(1, nQuant):
        borders = np.insert(borders, i, int((256 / nQuant) * i))
    histOrig, binsOrig = np.histogram(img.flatten(), bins=255, range=(0, 255))
    MSE_error_list = list()
    quantized_image_list = list()
    colourArray = np.zeros(nQuant)
    # Performing the quantization process steps nIter times:
    for i in range(nIter):
        # Taking from the origin hist and bins the values according to the borders,
        # and for each segment we calculate the weighted mean.
        for j in range(len(borders)-1):
            bins = binsOrig[borders[j].astype(int):borders[j+1].astype(int)]
            hist = histOrig[borders[j].astype(int):borders[j+1].astype(int)]
            weightedMean = (hist.flatten() * bins.flatten()).sum() / hist.sum()
            if np.isnan(weightedMean):
                weightedMean = np.nan_to_num(weightedMean)
            colourArray[j] = weightedMean
        # Finding the new borders
        for j in range(1, len(borders) - 1):
            borders[j] = (colourArray[j - 1] + colourArray[j]) / 2
        # Setting the new colours
        newImg = np.zeros(img.shape)
        for j in range(len(borders) - 1):
            newImg[(img < borders[j + 1]) & (img >= borders[j])] = borders[j]
        # Calculating the MSE => Mean Squared Error
        MSE = np.sqrt(np.power(np.subtract(newImg, img), 2).sum()) / (imOrig.shape[0] * imOrig.shape[1])
        MSE_error_list.append(MSE)
        quantized_image_list.append(newImg.copy())
    # Transforming back to RGB from YIQ after we took the Y channel
    if len(imOrig.shape) == 3:
        YIQ[:, :, 0] = newImg
        newImg = transformYIQ2RGB(YIQ) / 255
        plt.imshow(newImg)
        plt.show()
        plt.plot(MSE_error_list)
        plt.show()
    # Image is Grayscale:
    else:
        plt.gray()
        plt.imshow(newImg)
        plt.show()
        plt.plot(MSE_error_list)
        plt.show()

    return quantized_image_list, MSE_error_list


