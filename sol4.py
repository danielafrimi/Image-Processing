# ~~~~~~~~~~~~~~~~~~~~~      EX4      ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ Daniel Afrimi ~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy import ndimage
import shutil
from scipy.misc import imsave
import sol4_utils


def get_response(matrix_m):
    """
  Finding R for every pixel results in a response image R
 :param matrix_m:
 :return:
 """
    im_response = np.linalg.det(matrix_m) - (0.04) * np.power(
        np.trace(matrix_m, axis1=2, axis2=3), 2)
    return im_response


def get_derivatives(im):
    """
 :param im:
 :return:
 """
    filter_dev = np.array([[1, 0, -1]])
    dev_x = signal.convolve2d(im, filter_dev, mode='same', boundary='symm')
    dev_y = signal.convolve2d(im, filter_dev.T, mode='same',
                              boundary='symm')
    return dev_x, dev_y


def harris_corner_detector(im):
    """
 Detects harris corners.
 :param im: A 2D array representing an image.
 :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
 """
    dev_x, dev_y = get_derivatives(im)
    blur_dev_x_squre, blur_dev_y_squre, blur_dev_xy = blur_images(dev_x,
                                                                  dev_y)

    # array that in each place there is a marix 2x2 M that corresponds to the original pixles.
    m_matrix = np.empty((im.shape[0], im.shape[1], 2, 2))
    m_matrix[:, :, 0, 0] = blur_dev_x_squre
    m_matrix[:, :, 1, 1] = blur_dev_y_squre
    m_matrix[:, :, 0, 1] = blur_dev_xy
    m_matrix[:, :, 1, 0] = blur_dev_xy
    im_response = get_response(m_matrix)
    binary_image = non_maximum_suppression(im_response)

    # Merge the arrays of cooridinats that are not a zero (corners) and change the array shape
    corners = np.dstack(np.nonzero(binary_image))[0]
    return np.flip(corners, 1)


def sample_descriptor(im, pos, desc_rad):
    """
 Samples descriptors at the given corners.
 :param im: A 2D array representing an image.
 :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
 :param desc_rad: "Radius" of descriptors to compute.
 :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
 """
    size_patch = 1 + 2 * desc_rad
    desciptor = np.empty((pos.shape[0], size_patch, size_patch),
                         dtype=im.dtype)
    for i in range(len(pos)):
        # Creating the patch for a specific position
        position_x = np.arange(pos[i, 0] - desc_rad,
                               pos[i, 0] + desc_rad + 1)
        position_y = np.arange(pos[i, 1] - desc_rad,
                               pos[i, 1] + desc_rad + 1)

        # Create a rectangular grid out of an array of x values and an array of y values
        grid = np.meshgrid(position_y, position_x, indexing='ij')

        # Interpolating the points in each patch
        patch = ndimage.map_coordinates(im, grid, order=1, prefilter=False)

        # Normalize the Patch for making the descriptor invariant to certain changes of lighting.
        norm_patch = np.linalg.norm((patch - np.mean(patch)))
        if norm_patch != 0:
            desciptor[i, :, :] = (patch - np.mean(patch)) / norm_patch
        else:
            desciptor[i, :, :] = (patch - np.mean(patch))

    return desciptor


def find_features(pyr):
    """
 Detects and extracts feature points from a pyramid.
 :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
 :return: A list containing:
             1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                These coordinates are provided at the pyramid level pyr[0].
             2) A feature descriptor array with shape (N,K,K)
 """
    position = spread_out_corners(pyr[0], 7, 7, 3 * 2 ** 2)

    # We get to the third level in the pyramid - > pos/2/2 = pos/4
    # Sampling descriptors for the feature points
    desciptor = sample_descriptor(pyr[2], position / 4, 3)
    return position, desciptor


def match_features(desc1, desc2, min_score):
    """
 Return indices of matching descriptors.
 :param desc1: A feature descriptor array with shape (N1,K,K).
 :param desc2: A feature descriptor array with shape (N2,K,K).
 :param min_score: Minimal match score.
 :return: A list containing:
             1) An array with shape (M,) and dtype int of matching indices in desc1.
             2) An array with shape (M,) and dtype int of matching indices in desc2.
 """

    n1, n2, k = desc1.shape[0], desc2.shape[0], desc1.shape[2]

    # Flatten the Descriptors arrays, each row is a descriptor for some
    #  feature (insted of desciptor matrix for a feature of size k x k we a vector size of k^2)
    desc1_reshape = desc1.reshape((n1, k * k))
    desc2_reshape = desc2.reshape((n2, k * k))

    # Getting an array of scores between the descriptor size of n1 x n2
    score = np.dot(desc1_reshape, desc2_reshape.transpose())

    # Takes the biggest 2 in col and row (sort the two last element - biggest)
    partition_row = np.partition(score, -2, axis=1)
    partition_col = np.partition(score, -2, axis=0)

    max_two_col = partition_col[-2, :].reshape(1, n2)
    max_two_row = partition_row[:, -2].reshape(n1, 1)

    # Checks the conditions
    match1, match2 = np.where(
        (score >= max_two_col) & (score >= max_two_row) & (
            score > min_score))
    return match1, match2


def apply_homography(pos1, H12):
    """
 Apply homography to inhomogenous points.
 :param pos1: An array with shape (N,2) of [x,y] point coordinates.
 :param H12: A 3x3 homography matrix.
 :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
 """

    N, col = pos1.shape[0], pos1.shape[1]
    # Create a new pos1 with the value 1 in the col
    pos1_add_col = np.zeros((N, col + 1))
    pos1_add_col[:, :-1] = pos1
    pos1_add_col[:, -1] = 1

    # Apply dot product between the homography matrix to the cooridinates
    matrix_transformation = np.dot(H12, pos1_add_col.transpose())
    # Divide in the last cooridinate
    matrix_transformation = matrix_transformation / matrix_transformation[
                                                    -1, :]

    matrix_transformation = matrix_transformation.transpose()

    # Cut the last col for shape (N,2)
    return matrix_transformation[:, :-1]


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    """
 Computes homography between two sets of points using RANSAC.
 :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
 :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
 :param num_iter: Number of RANSAC iterations to perform.
 :param inlier_tol: inlier tolerance threshold.
 :param translation_only: see estimate rigid transform
 :return: A list containing:
             1) A 3x3 normalized homography matrix.
             2) An Array with shape (S,) where S is the number of inliers,
                 containing the indices in pos1/pos2 of the maximal set of inlier matches found.
 """
    # RANSAC Algorithm:

    # 1. Sample (randomly) the number of points required to fit the model
    # (in case of translation sample 1 point and in case of rigid sample 2 points)

    # 2. Solve for model parameters using samples

    # 3. Score by the fraction of inliers within a preset threshold of the model

    # Repeat num_iter until the best model is found with high confidence

    if num_iter == 0 or len(points1) == 0 or len(points2) == 0:
        return np.eye(3), []

    # Array of size N
    arr_indices = np.arange(points1.shape[0])

    # Set of inliers
    max_set = np.array([])

    param = 1
    if not translation_only:
        param = 2

    # Runing n_iter iteration when we random 2 pair points from both points
    # (diffrent images), compute the rigid transforming + error of the transfom and selecting inliers
    for i in range(num_iter):
        # Shuffle the Array - Stage 1 in RANSAC Algorithm
        np.random.shuffle(arr_indices)
        # Takes random Points according to param
        param_inidices = arr_indices[:param]
        set_indices_points1 = points1[param_inidices]
        set_indices_points2 = points2[param_inidices]

        # Matrix 3 x 3 --> Computes rigid transforming points1 towards points2, using least squares method.
        # Stage 2 - RANSAC Algorithm
        H12 = estimate_rigid_transform(set_indices_points1,
                                       set_indices_points2,
                                       translation_only)

        if H12 is not None:

            # Compute the transforming points1 according the H12 that computed before
            points2_transform = apply_homography(points1, H12)

            # Compute the error - Eucldian distance (squre)
            # Stage 3 - RANSAC Algorithm
            error = np.power(
                np.linalg.norm(points2_transform - points2, axis=1), 2)
            # Marks all matches having Error < inlier_tol as inlier matches
            inliers = np.nonzero(error < inlier_tol)[0]
            if (inliers.shape[0] > max_set.shape[0]):
                max_set = inliers

    H12 = estimate_rigid_transform(points1[max_set], points2[max_set],
                                   translation_only)

    return H12, max_set


def display_matches(im1, im2, points1, points2, inliers):
    """
 Dispalay matching points.
 :param im1: A grayscale image.
 :param im2: A grayscale image.
 :param pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
 :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
 :param inliers: An array with shape (S,) of inlier matches.
 """
    combine_images = np.hstack((im1, im2))
    plt.imshow(combine_images, cmap='gray')

    # Creates yellow and blue lines according to inliers and outliers
    for i in range(points2.shape[0]):
        arr_cooridinate_x = np.array(
            [points1[i, 0], points2[i, 0] + im1.shape[1]])
        arr_cooridinate_y = np.array([points1[i, 1], points2[i, 1]])
        # Check if its outlier on inlier
        if i not in inliers:
            plt.plot(arr_cooridinate_x, arr_cooridinate_y, mfc='r', c='b',
                     lw=.4, ms=2, marker='o')
        else:
            plt.plot(arr_cooridinate_x, arr_cooridinate_y, mfc='r', c='y',
                     lw=.4, ms=2, marker='o')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """

    H2m_return = [np.eye(H_succesive[0].shape[0])]
    M = len(H_succesive)
    # i < m
    for i in range(m - 1, -1, -1):
        H = np.dot(H2m_return[0], H_succesive[i])
        H2m_return.insert(0, np.divide(H, H[2, 2]))

    # i > m
    for i in range(m, M):
        inverse_H = np.linalg.inv(H_succesive[i])
        # For images that come after the refrence image we should use the inverse Homography
        H = np.dot(H2m_return[-1], inverse_H)
        H2m_return.append(np.divide(H, H[2, 2]))

    return H2m_return


def compute_bounding_box(homography, w, h):
    """
 computes bounding box of warped image under homography, without actually warping the image
 :param homography: homography
 :param w: width of the image
 :param h: height of the image
 :return: 2x2 array, where the first row is [x,y] of the top left corner,
  and the second row is the [x,y] of the bottom right corner
 """
    # top-left, top-right, bottom-right, bottom-left
    corners_image = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    position_homography = apply_homography(corners_image, homography)

    # top left corner
    tl = np.array([np.min(position_homography[:, 0]),
                   np.min(position_homography[:, 1])])
    # bottom right corner
    bt = np.array([np.max(position_homography[:, 0]),
                   np.max(position_homography[:, 1])])
    return np.array([tl, bt]).astype(np.int)


def warp_channel(image, homography):
    """
 Warps a 2D image with a given homography.
 :param image: a 2D image.
 :param homography: homograhpy.
 :return: A 2d warped image.
 """
    rows, col = image.shape

    # Gets the upper left corner and the lower right + wrap image according to this homography
    corners = compute_bounding_box(homography, col, rows)
    x_min, x_max = corners[0, 0], corners[1, 0]
    y_min, y_max = corners[0, 1], corners[1, 1]

    # Takes the corners
    m, n = y_max - y_min + 1, x_max - x_min + 1
    x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max + 1),
                                     np.arange(y_min, y_max + 1))
    inverse_homography = np.linalg.inv(homography)
    inverse_homography /= inverse_homography[2, 2]
    points = np.concatenate(
        (x_coords.reshape(m * n, 1), y_coords.reshape(m * n, 1)), axis=1)

    # Back Wraping
    points_two = apply_homography(points, inverse_homography)
    coordinates = [points_two[:, 1], points_two[:, 0]]
    # interpolation for the pints - give them the correct intensity
    interpolated = ndimage.map_coordinates(image, coordinates, order=1,
                                           prefilter=False)
    warped_image = interpolated.reshape(m, n)
    return warped_image


def warp_image(image, homography):
    """
 Warps an RGB image with a given homography.
 :param image: an RGB image.
 :param homography: homograhpy.
 :return: A warped image.
 """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])


def filter_homographies_with_translation(homographies,
                                         minimum_right_translation):
    """
 Filters rigid transformations encoded as homographies by the amount of translation from left to right.
 :param homographies: homograhpies to filter.
 :param minimum_right_translation: amount of translation below which the transformation is discarded.
 :return: filtered homographies..
 """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
 Computes rigid transforming points1 towards points2, using least squares method.
 points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
 :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
 :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
 :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
 :return: A 3x3 array with the computed homography.
 """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def blur_images(dev_x, dev_y):
    """
 :param dev_x:
 :param dev_y:
 :return:
 """
    blur_dev_x_squre = sol4_utils.blur_spatial((dev_x * dev_x), 3)
    blur_dev_y_squre = sol4_utils.blur_spatial((dev_y * dev_y), 3)
    dxy = np.multiply(dev_x, dev_y)
    blur_dev_xy = sol4_utils.blur_spatial(dxy, 3)
    return blur_dev_x_squre, blur_dev_y_squre, blur_dev_xy


def non_maximum_suppression(image):
    """
 Finds local maximas of an image.
 :param image: A 2D array representing an image.
 :return: A boolean array with the same shape as the input image, where True indicates local maximum.
 """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
 Splits the image im to m by n rectangles and uses harris_corner_detector on each.
 :param im: A 2D array representing an image.
 :param m: Vertical number of rectangles.
 :param n: Horizontal number of rectangles.
 :param radius: Minimal distance of corner points from the boundary of the image.
 :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
 """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1],
                     x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,
                           :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = (
        (corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
        (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
      The naming convention for a sequence of images is file_prefixN.jpg,
      where N is a running number 001, 002, 003...
      :param data_dir: path to input images.
      :param file_prefix: see above.
      :param num_images: number of images to produce the panoramas with.
      """
        self.file_prefix = file_prefix
        self.files = [
            os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for
            i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        self.bonus = bonus
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
      compute homographies between all images to a common coordinate system
      :param translation_only: see estimte_rigid_transform
      """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (
            len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros(
            (self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(
                self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2,
                                    endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros(
            (number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[
                              None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in
                              self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :,
                                      0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = (
            (warped_slice_centers[:, :-1] + warped_slice_centers[:,
                                            1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) *
                                      panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros(
            (number_of_panoramas, panorama_size[1], panorama_size[0], 3),
            dtype=np.float64)
        if self.bonus == False:
            for i, frame_index in enumerate(self.frames_for_panoramas):
                # warp every input image once, and populate all panoramas
                image = sol4_utils.read_image(self.files[frame_index], 2)
                warped_image = warp_image(image, self.homographies[i])
                x_offset, y_offset = self.bounding_boxes[i][0].astype(
                    np.int)
                y_bottom = y_offset + warped_image.shape[0]

                for panorama_index in range(number_of_panoramas):
                    # take strip of warped image and paste to current panorama
                    boundaries = x_strip_boundary[panorama_index, i:i + 2]
                    image_strip = warped_image[:,
                                  boundaries[0] - x_offset: boundaries[
                                                                1] - x_offset]
                    x_end = boundaries[0] + image_strip.shape[1]
                    self.panoramas[panorama_index, y_offset:y_bottom,
                    boundaries[0]:x_end] = image_strip

            # crop out areas not recorded from enough angles
            # assert will fail if there is overlap in field of view between the left most image and the right most image
            crop_left = int(self.bounding_boxes[0][1, 0])
            crop_right = int(self.bounding_boxes[-1][0, 0])
            assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
            print(crop_left, crop_right)
            self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


        else:
            for i, frame_index in enumerate(self.frames_for_panoramas):
                image = sol4_utils.read_image(self.files[frame_index], 2)
                warped_image = warp_image(image, self.homographies[i])
                x_offset, y_offset = self.bounding_boxes[i][0].astype(
                    np.int)
                y_bottom = y_offset + warped_image.shape[0]

                for panorama_index in range(number_of_panoramas):
                    boundaries = x_strip_boundary[panorama_index, i:i + 2]
                    image_strip = warped_image[:,
                                  boundaries[0] - x_offset: boundaries[
                                                                1] - x_offset]

                    x_end = boundaries[0] + image_strip.shape[1]
                    if i == 0:
                        self.panoramas[panorama_index, y_offset:y_bottom,
                        boundaries[0]:x_end] = image_strip
                    if i > 0:
                        max_h = min(warped_image.shape[0],
                                    last_warped.shape[0])
                        max_h = max_h - (max_h % 16)

                        # creating mask - half frame is ones and the another frame is zeros
                        mask = np.column_stack((np.ones(shape=(max_h, 32)),
                                                np.zeros(
                                                    shape=(max_h, 32))))

                        blending_strip_1 = warped_image[:max_h,
                                           boundaries[0] - x_offset - 32:
                                           boundaries[0] - x_offset + 32]
                        blending_strip_2 = last_warped[:max_h,
                                           boundaries[0] - x_offset - 32:
                                           boundaries[0] - x_offset + 32]

                        # Blending the strips with the relevant func from ex3 - according to RGB
                        blending_strip = self.blending_strip_func(
                            blending_strip_1, blending_strip_2, mask)
                        zero_strip = np.zeros(
                            (image_strip.shape[0], 64, 3))
                        zero_strip[:max_h, :, :] = blending_strip

                        # stitch the blending strip
                        self.panoramas[panorama_index, y_offset:y_bottom,
                        boundaries[0]:x_end] = image_strip
                        self.panoramas[panorama_index, y_offset:y_bottom,
                        boundaries[0] - 32: boundaries[
                                                0] + 32] = zero_strip

                last_warped = warped_image

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def blending_strip_func(self, blending_strip_1, blending_strip_2,
                            mask):
        blending_strip = np.zeros(blending_strip_1.shape)
        blending_strip[:, :, 0] = sol4_utils.pyramid_blending(
            blending_strip_1[:, :, 0], blending_strip_2[:, :, 0], mask, 4,
            9, 9)
        blending_strip[:, :, 1] = sol4_utils.pyramid_blending(
            blending_strip_1[:, :, 1], blending_strip_2[:, :, 1], mask, 4,
            9, 9)
        blending_strip[:, :, 2] = sol4_utils.pyramid_blending(
            blending_strip_1[:, :, 2], blending_strip_2[:, :, 2], mask, 4,
            9, 9)
        return blending_strip

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()