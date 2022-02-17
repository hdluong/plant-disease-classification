from rembg.bg import remove
import numpy as np
import io
import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

images_source = 'input/'
images_output = 'output/'

if __name__ == '__main__':
    # folder
    if os.path.isdir(images_source):
        # read files
        files = [entry for entry in os.listdir(images_source)
                if os.path.isfile(os.path.join(images_source, entry))]   

        for i, file in enumerate(files):
            if (file.endswith('.JPG')):
                filename, ext = os.path.splitext(file)

                filePath = np.fromfile(os.path.join(images_source, file))
                # read image and segment
                result = remove(filePath)
                # handle ouput folder and fileaname
                new_filename = filename + '_marked' + ext
                new_filename = os.path.join(images_output, new_filename)

                img = Image.open(io.BytesIO(result)).convert("RGB")
                img.save(new_filename)