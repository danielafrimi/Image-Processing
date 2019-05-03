from scipy.misc import imread
import numpy as np
from skimage.color import rgb2gray
from scipy import signal
import scipy.ndimage.filters as filters

def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representaion: representation code, either 1 or 2 defining
     whether the output should be a grayscaleimage (1) or an RGB image (2)
    :return: image
    """
    if representation != 1 and representation != 2:
        raise Exception("bad representation")
    im = imread(filename).astype(np.float64)
    im /= 255
    if representation == 1:
        im = rgb2gray(im)
        return im
    return im

def blur_spatial(im, kernel_size):
    """
    performs image blurring using 2D convolution between the image f and a gaussian kernel g.
    :param im: is the input image to be blurred (grayscale float64 image).
    :param kernel_size: is the size of the gaussian kernel in each dimension (an odd integer).
    :return: The function returns the output blurry image (grayscale float64 image).
    """

    return signal.convolve2d(im, create_gaussian_2D(kernel_size), mode='same')


def create_gaussian_2D(kernel_size):
    """
    Binomial coefficients provide a	compact approximation of the gaussian coefficients using only integers.
    the simplest blur filter is [1,1].
    :return: gaussian_2D kernel
    """
    if kernel_size == 1:
        return np.array([[1]])
    binomi_base1D = np.array([[1, 1]]).astype(np.float64)
    gaussian_copy_base = binomi_base1D.copy().astype(np.float64)
    num_iter = binomi_base1D.shape[1]
    while num_iter < kernel_size:
        # Convolotion with itself as number of the kernel size
        gaussian_copy_base = signal.convolve2d(gaussian_copy_base, binomi_base1D)
        num_iter += 1


    # Makes Matrix - filter
    gaussian_2D = np.outer(gaussian_copy_base, gaussian_copy_base)

    # Make sure the sum is 1
    gaussian_2D = gaussian_2D/np.sum(gaussian_2D)
    return gaussian_2D



def build_gaussian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1]
    (e.g. the output of ex1’s read_image with the representationset to 1).
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
    in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return:1. pyramid pyr as a standard python array (i.e. not numpy’s array)
     with maximum length of max_levels, where each element of the array is a grayscale image.
     2. filter_vec which is row vector of shape (1, filter_size) used
     for the pyramid construction. This filter should be built using a consequent 1D convolutions of [1 1]
     with itself in order to derive a row of the binomial coefficients which is a good approximation to
     the Gaussian profile. The filter_vec should be normalized.
    """
    pyr = []
    pyr.append(im)
    filter_vec = create_gaussian_1D(filter_size).reshape((1, filter_size))
    for i in range(1, max_levels):
        reduced_im = reduce(pyr[-1], filter_vec)
        pyr.append(reduced_im)

        # checks if dimension is good, if the condition exist the next iteration the dim will be less than 16
        if min(reduced_im.shape[0], reduced_im.shape[1]) < 32: # todo change ?
            return pyr, filter_vec

    return pyr, filter_vec


def create_gaussian_1D(kernel_size):
    """
     Binomial coefficients provide a	compact approximation of the gaussian coefficients using only integers.
     the simplest blur filter is [1,1].
     :return: gaussian_2D kernel
     """
    if kernel_size == 1:
        return np.array([[1]])
    binomi_base1D = np.array([1, 1], dtype=np.uint64)
    gaussian_copy_base = binomi_base1D.copy()
    for i in range(kernel_size - 2):
        gaussian_copy_base = signal.convolve(gaussian_copy_base, binomi_base1D)
    gaussian = gaussian_copy_base / np.sum(gaussian_copy_base)
    return gaussian



def expand_im(im, filter_vec):
    """

    :param im:
    :param filter_vec:
    :return:
    """
    new_shape_row = 2*im.shape[0]
    new_shape_col =2*im.shape[1]
    expand_im_zeroes = np.zeros((new_shape_row, new_shape_col), dtype=im.dtype)
    expand_im_zeroes[::2, ::2] = im
    convolved_row = filter_vec.reshape((filter_vec.size, 1)) * 2
    convolved_col = convolved_row.transpose()
    blur = filters.convolve(expand_im_zeroes, convolved_row)
    blur = filters.convolve(blur, convolved_col)
    return blur


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: which is row vector of shape (1, filter_size) used
     for the pyramid construction. This filter should be built using a consequent 1D convolutions of [1 1]
     with itself in order to derive a row of the binomial coefficients which is a good approximation to
     the Gaussian profile. The filter_vec should be normalized.
    :param coeff: python list. The list length is the same as the number of levels in the pyramid lpyr.
    :return:
    """
    lpyrCoeff = lpyr * np.array(coeff)
    to_im = lpyrCoeff[-1]

    # pass all the L_n from top to bottom
    for i in range(len(lpyr)-1, 0 , -1):
        expanded_im = expand_im(to_im, filter_vec)
        to_im = expanded_im + lpyrCoeff[i - 1]

    return to_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    """

    :param im: a grayscale image with double values in [0, 1]
    (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
    (an odd scalar that represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return:1. pyramid pyr as a standard python array (i.e. not numpy’s array)
     with maximum length of max_levels, where each element of the array is a grayscale image.
     2. filter_vec which is row vector of shape (1, filter_size) used
     for the pyramid construction. This filter should be built using a consequent 1D convolutions of [1 1]
     with itself in order to derive a row of the binomial coefficients which is a good approximation to
     the Gaussian profile. The filter_vec should be normalized.
    """
    pyr = []
    pyr_gaussian, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range (1, len(pyr_gaussian)):
        expanded_im = expand_im(pyr_gaussian[i], filter_vec)
        substract_im = pyr_gaussian[i-1] - expanded_im
        pyr.append(substract_im)

    pyr.append(pyr_gaussian[-1])
    return pyr, filter_vec


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """

    :param im1: grayscale image to be blended.
    :param im2: grayscale image to be blended.
    :param mask: is a boolean mask containing True and False representing which parts
    of im1 and im2 should appear in the resulting im_blend.
    :param max_levels: – is the max_levels parameter use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter which defining the filter used
    in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: – is the size of the Gaussian filter which
    defining the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    """
    lap_im1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_im2 , filter_vec2= build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask = mask.astype(np.float64)
    mask_gauss , filter_vec3 = build_gaussian_pyramid(mask, max_levels, filter_size_mask)

    laplacian_out = []
    for i in range(len(mask_gauss)):
        laplacian_out.append((mask_gauss[i]*lap_im1[i]) + ((1-mask_gauss[i])* lap_im2[i]))

    coeff = [1]*len(laplacian_out)
    im_blend = np.clip(laplacian_to_image(laplacian_out, filter_vec1, coeff), 0, 1)
    return im_blend



def reduce(im, filter_vec):
    """

    :param im:
    :param filter_vec:
    :return:
    """
    # Convolotion on both axis (row and col)

    blur_image = filters.convolve(im, filter_vec.reshape((filter_vec.size, 1)))
    blur_image = filters.convolve(blur_image, filter_vec.reshape((filter_vec.size, 1)).transpose())

    # taking only the even indices to resize the image
    return blur_image[::2, ::2]






