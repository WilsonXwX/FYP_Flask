
from skimage import exposure
import numpy as np
import argparse
import cv2
import sys
# from flask import Flask, request, jsoexposurenify, send_file
from flask import Flask, request, send_file
import io
import os
from os.path import exists
import os.path as pathfile
from PIL import Image

import sys
import os
sys.path.append(r'./')
# print(sys.path)
from src.colorspace import *
from src.io_ import *
from src.api import *
# import src
import numpy as np
# from skimage.color import colorconv
from src.color import *

def find_color_card(image):
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,
                                                       arucoDict, parameters=arucoParams)

    # try to extract the coordinates of the color correction card
    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()

        # extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]

        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]

        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]

        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]

    # we could not find color correction card, so gracefully return
    except:
        return None

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, bird’s-eye view of the color
    # matching card
    cardCoordinates = np.array([topLeft, topRight,
                           bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoordinates)
    # return the color matching card to the calling function
    return card


def _match_cumulative_cdf_mod(source, template, full):
    """
    Return modified full image array so that the cumulative density function of
    source array matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    # Here we compute values which the channel RGB value of full image will be modified to.
    interpb = []
    for i in range(0, 256):
        interpb.append(-1)

    # first compute which values in src image transform to and mark those values.

    for i in range(0, len(interp_a_values)):
        frm = src_values[i]
        to = interp_a_values[i]
        interpb[frm] = to

    # some of the pixel values might not be there in interp_a_values, interpolate those values using their
    # previous and next neighbours
    prev_value = -1
    prev_index = -1
    for i in range(0, 256):
        if interpb[i] == -1:
            next_index = -1
            next_value = -1
            for j in range(i + 1, 256):
                if interpb[j] >= 0:
                    next_value = interpb[j]
                    next_index = j
            if prev_index < 0:
                interpb[i] = (i + 1) * next_value / (next_index + 1)
            elif next_index < 0:
                interpb[i] = prev_value + ((255 - prev_value) * (i - prev_index) / (255 - prev_index))
            else:
                interpb[i] = prev_value + (i - prev_index) * (next_value - prev_value) / (next_index - prev_index)
        else:
            prev_value = interpb[i]
            prev_index = i

    # finally transform pixel values in full image using interpb interpolation values.
    wid = full.shape[1]
    hei = full.shape[0]
    ret2 = np.zeros((hei, wid))
    for i in range(0, hei):
        for j in range(0, wid):
            ret2[i][j] = interpb[full[i][j]]
    return ret2


def match_histograms_mod(inputCard, referenceCard, fullImage):
    """
        Return modified full image, by using histogram equalizatin on input and
         reference cards and applying that transformation on fullImage.
    """
    if inputCard.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')
    matched = np.empty(fullImage.shape, dtype=fullImage.dtype)
    for channel in range(inputCard.shape[-1]):
        matched_channel = _match_cumulative_cdf_mod(inputCard[..., channel], referenceCard[..., channel],
                                                    fullImage[..., channel])
        matched[..., channel] = matched_channel
    return matched

def test(filename, L=False, **kwargs):
    img = cv2.imread(filename)
    detector = cv2.mcc.CCheckerDetector_create()
    result = detector.process(img,0)
    list = detector.getListColorChecker()
    if not list:
        return False
    best_checker = detector.getBestColorChecker()
    result = np.array(best_checker.getChartsRGB())
    num_rows, num_cols = result.shape

    src_col = result[:, 1]
    src_col_reshaped = src_col.reshape(num_rows//3,3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    src = src_col_reshaped
    ccm = color_calibration(src/255, Macbeth_D65_2, **kwargs)

    img = ccm.infer_image(filename, L)
    cv2.imwrite('TestOutput.png', img)
    return True

app = Flask(__name__)

@app.route('/colorCorrection', methods=['POST'])
def colorCorrection():
    try:
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        

        # img1 = cv2.imread(request.files['image'])
        
        image_file = request.files['image']
        img = Image.open(io.BytesIO(image_file.read()))
        img.save("inputimage.png")
        
        # img1 = cv2.imread("inputimage.png")

        # img.save("inputimage.png")
        # img1 = cv2.imread("inputimage.png")

        if test('inputimage.png', colorspace = sRGB)==False:
            return jsonify({'error': 'No color card found'}), 400

        
        # result2 = match_histograms_mod(imageCard, rawCard, img1)

        # # Create a byte buffer to store the adjusted image
        # cv2.imwrite("output1.png", result2)
        # output_exists = pathfile.isfile("output1.png")
        # if not output_exists:
        #     return jsonify({'error': 'Could not process the image'}), 400

        # imageprocessed = Image.fromarray(result2)


        imageProcessed = Image.open('TestOutput.png')
        buffer = io.BytesIO()
        imageProcessed.save(buffer, format="png")
        buffer.seek(0)

        # os.remove("TestOutput.jpeg")

        return send_file(buffer, mimetype='image/png')
        # return jsonify({'error': 'YesYouHave error'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
