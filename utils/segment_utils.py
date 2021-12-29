import numpy as np
import cv2
import os

# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

# error message when image is not colored while it should be
NOT_COLOR_IMAGE = 'NOT_COLOR_IMAGE'

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.true_divide(a, b)
        q[ ~ np.isfinite(q) ] = 0  # -inf inf NaN

    return q

def read_image(file_path, read_mode=cv2.IMREAD_COLOR):
    """
    Read an image file from disk

    Args:
        file_path: absolute file_path of an image file
        read_mode: whether image reading mode is rgb, grayscale or somethin

    Returns:
        np.ndarray of the read image or None if couldn't read

    Raises:
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path, read_mode)

    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image

def index_diff(image, green_scale=2.0, red_scale=1.4):
    """
    Compute color indices to define non-green background

    args:
        image: numpy array of a color RGB image
        green_scale: number to scale green channel of the image
        red_scale: number to scale red channel of the image
    returns:
        a numpy array as the difference between these color indices
    """

    if len(image.shape) < 3:
        raise ValueError(NOT_COLOR_IMAGE)

    bgr_sum = np.sum(image, axis=2)

    blues = div0(image[:, :, 0], bgr_sum)
    greens = div0(image[:, :, 1], bgr_sum)
    reds = div0(image[:, :, 2], bgr_sum)

    green_index = green_scale * greens - (reds + blues)
    red_index = red_scale * reds - (greens)

    return green_index - red_index

# segment methods
METHODS = {
    'HSV': 0,
    'COLOR_DIFF': 1
}

def generate_floodfill_mask(bin_image):
    """
    Generate a mask to remove backgrounds adjacent to image edge

    Args:
        bin_image (ndarray of grayscale image): image to remove backgrounds from

    Returns:
        a mask to backgrounds adjacent to image edge
    """
    y_mask = np.full(
        (bin_image.shape[0], bin_image.shape[1]), fill_value=255, dtype=np.uint8
    )
    x_mask = np.full(
        (bin_image.shape[0], bin_image.shape[1]), fill_value=255, dtype=np.uint8
    )

    xs, ys = bin_image.shape[0], bin_image.shape[1]

    for x in range(0, xs):
        item_indexes = np.where(bin_image[x,:] != 0)[0]

        if len(item_indexes):
            start_edge, final_edge = item_indexes[0], item_indexes[-1]
            x_mask[x, start_edge:final_edge] = 0

    for y in range(0, ys):
        item_indexes = np.where(bin_image[:,y] != 0)[0]
        if len(item_indexes):
            start_edge, final_edge = item_indexes[0], item_indexes[-1]

            y_mask[start_edge:final_edge, y] = 0

    return np.logical_or(x_mask, y_mask)

def color_index_marker(color_index_diff, marker):
    """
    Differentiate marker based on the difference of the color indexes
    Threshold below some number(found empirically based on testing on 5 photos,bad)
    If threshold number is getting less, more non-green image
     will be included and vice versa
    Args:
        color_index_diff: color index difference based on green index minus red index
        marker: marker to be updated

    Returns:
        nothing
    """
    marker[color_index_diff <= -0.05] = False

def generate_background_marker(file):
    """
    Generate background marker for an image

    Args:
        file (string): full path of an image file

    Returns:
        tuple[0] (ndarray of an image): original image
        tuple[1] (ndarray size of an image): background marker
    """

    # check file name validity
    if not os.path.isfile(file):
        raise ValueError('{}: is not a file'.format(file))

    original_image = read_image(file)

    marker = np.full((original_image.shape[0], original_image.shape[1]), True)

    # update marker based on vegetation color index technique
    color_index_marker(index_diff(original_image), marker)

    return original_image, marker

def select_largest_obj(img_bin, lab_val=255, smooth_boundary=False, kernel_size=15):
    """
    Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin (2D array): 2D numpy array of binary image.
        lab_val ([int]): integer value used for the label of the largest
                object. Default is 255.
        smooth_boundary ([boolean]): whether smooth the boundary of the
                largest object using morphological opening or not. Default
                is false.
        kernel_size ([int]): the size of the kernel used for morphological
                operation. Default is 15.
    Returns:
        a binary image as a mask for the largest object.
    """

    # set up components
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)

    # find largest component label(label number works with labeled image because of +1)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1

    # create a mask that will only cover the largest component
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    
    # fill holes using opencv floodfill function
    # set up seedpoint(starting point) for floodfill
    bkg_locs = np.where(img_labeled == 0)
    bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])

    # copied image to be floodfill
    img_floodfill = largest_mask.copy()

    # create a mask to ignore what shouldn't be filled(I think no effect)
    h_, w_ = largest_mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)

    cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                newVal=lab_val)
    holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.

    # get a mask to avoid filling non-holes that are adjacent to image edge
    non_holes_mask = generate_floodfill_mask(largest_mask)
    holes_mask = np.bitwise_and(holes_mask, np.bitwise_not(non_holes_mask))

    largest_mask = largest_mask + holes_mask
    
    if smooth_boundary:
        # smooth edge boundary
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN,
                                        kernel_)

    return largest_mask

def segment_color_diff(image_file):
    """
    Segments leaf from an image file

    Args:
        image_file (string): full path of an image file

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
    """
    # get background marker and original image
    original, marker = generate_background_marker(image_file)

    # set up binary image for futher processing
    bin_image = np.zeros((original.shape[0], original.shape[1]))
    bin_image[marker] = 255
    bin_image = bin_image.astype(np.uint8)

    # further processing of image, filling holes, smoothing edges
    largest_mask = select_largest_obj(bin_image)
    # apply marker to original image
    result = original.copy()
    result[largest_mask == 0] = np.array([0, 0, 0])

    return original, result

def segment_hsv(image_file):
    """
    Segments leaf from an image file based on hsv

    Args:
        image_file (string): full path of an image file

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
    """

    original = read_image(image_file)
    # create hsv
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    # set lower and upper color limits
    low_val = (0,60,0)
    high_val = (179,255,255)
    # Threshold the HSV image 
    mask = cv2.inRange(hsv, low_val,high_val)
    
    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))

    # filling holes using floodfill
    # copied image to be floodfill
    img_floodfill = mask.copy()

    # create a mask to ignore what shouldn't be filled(I think no effect)
    h_, w_ = mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)

    cv2.floodFill(img_floodfill, mask_, seedPoint=(0,0),
                newVal=255)
    # mask of the holes.
    holes_mask = cv2.bitwise_not(img_floodfill)
    # get a mask to avoid filling non-holes that are adjacent to image edge
    non_holes_mask = generate_floodfill_mask(mask)
    holes_mask = np.bitwise_and(holes_mask, np.bitwise_not(non_holes_mask))

    mask_out = mask + holes_mask

    # apply mask to original image
    result = cv2.bitwise_and(original, original, mask=mask_out)
    
    return original, result
