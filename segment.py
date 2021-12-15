import argparse
import os
from utils.segment_utils import *

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-method', '--method', choices=['hsv', 'color_diff'], help='Name of segment method to use', default='hsv')
    ap.add_argument('-d', '--destination',
                        help='Destination directory for output image. '
                             'If not specified destination directory will be input image directory')
    ap.add_argument('-i', '--image_source', help='A path of image filename or folder containing images')
    
    # set up command line arguments conveniently
    args = vars(ap.parse_args())
    method = METHODS[args['method'].upper()]
    if args['destination']:
        if not os.path.isdir(args['destination']):
            print(args['destination'], ': is not a directory') 
            exit()

    # set up files to be segmented and destination place for segmented output
    if os.path.isdir(args['image_source']):
        files = [entry for entry in os.listdir(args['image_source'])
                 if os.path.isfile(os.path.join(args['image_source'], entry))]
        base_folder = args['image_source']

        # set up destination folder for segmented output
        if args['destination']:
            destination = args['destination']
        else:
            if args['image_source'].endswith(os.path.sep):
                args['image_source'] = args['image_source'][:-1]
            destination = args['image_source'] + '_markers'
            os.makedirs(destination, exist_ok=True)
    else:
        folder, file = os.path.split(args['image_source'])
        files = [file]
        base_folder = folder

        # set up destination folder for segmented output
        if args['destination']:
            destination = args['destination']
        else:
            destination = folder

    for i, file in enumerate(files):
        try:
            # read image and segment leaf
            if method == METHODS['HSV']:
                original, output_image = segment_hsv(os.path.join(base_folder, file))
            elif method == METHODS['COLOR_DIFF']:
                original, output_image = segment_color_diff(os.path.join(base_folder, file))

        except ValueError as err:
            if str(err) == IMAGE_NOT_READ:
                print('Error: Could not read image file: ', file)
            elif str(err) == NOT_COLOR_IMAGE:
                print('Error: Not color image file: ', file)
            else:
                raise
        # write segmented output
        else:
            # handle destination folder and fileaname
            filename, ext = os.path.splitext(file)
            new_filename = filename + '_marked' + ext
            new_filename = os.path.join(destination, new_filename)

            # write the output
            cv2.imwrite(new_filename, output_image)
            print('[INFO] {}/{} Marker generated for image file: {}'.format(i+1, len(files), file))
