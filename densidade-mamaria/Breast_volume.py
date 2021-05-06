import cv2
import numpy as np
import pydicom as dicom
import os
from matplotlib import pyplot
import itk
import sys
from skimage import exposure
import imutils
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist
import math

# Save image in set directory
# Read images
PathDicom = r"C:\Users\alex3\OneDrive - Universidade do Porto\Tese\densidade-mamaria"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName, filename))

# Select file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))

# plot
pyplot.imshow(RefDs.pixel_array, cmap=pyplot.cm.gray), pyplot.title('Original Image')
pyplot.show()

# Gaussian Filter
dcm_sample = RefDs.pixel_array

blur = cv2.GaussianBlur(dcm_sample, (5, 5), 0)

# Show pre-processing
pyplot.subplot(121), pyplot.imshow(dcm_sample, cmap=pyplot.cm.bone), pyplot.title('Original')
pyplot.xticks([]), pyplot.yticks([])
pyplot.subplot(122), pyplot.imshow(blur, cmap=pyplot.cm.bone), pyplot.title('Blurred')
pyplot.xticks([]), pyplot.yticks([])
pyplot.show()

# Convert to binary
(thresh, im_bw) = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im_bw = im_bw.astype(np.uint8)  # transform from 16bit to 8 bit

# Show binary
pyplot.imshow(im_bw, cmap=pyplot.cm.bone), pyplot.title('Binary')
pyplot.show()
cv2.imwrite('im_bw.tif', im_bw)

# remove artifact
kernel = np.ones((200, 150), np.uint8)
im_bw = cv2.dilate(im_bw, kernel, iterations=1)
im_bw = cv2.erode(im_bw, kernel, iterations=1)

# im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)

# Show binary without artifact
pyplot.imshow(im_bw), pyplot.title('Binary without artifact')
pyplot.show()

# perform edge detection, then perform a dilation + erosion to
# open gaps
edged = cv2.Canny(im_bw, 0, 255)
edged = cv2.dilate(edged, None, iterations=2)
edged = cv2.erode(edged, None, iterations=2)

pyplot.imshow(edged), pyplot.title('Edge')
pyplot.show()

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

# change image type
im_bw = blur

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    print("Contour area:", cv2.contourArea(c))
    if cv2.contourArea(c) < 100:
        continue
    # compute the rotated bounding box of the contour
    # orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(im_bw, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(im_bw, (int(x), int(y)), 5, (0, 0, 255), -1)

pyplot.imshow(im_bw), pyplot.title('contour')
pyplot.show()


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# unpack the ordered bounding box, then compute the midpoint
# between the top-left and top-right coordinates, followed by
# the midpoint between bottom-left and bottom-right coordinates
(tl, tr, br, bl) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)

# compute the midpoint between the top-left and top-right points,
# followed by the midpoint between the top-righ and bottom-right
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)
# draw the midpoints on the image
cv2.circle(im_bw, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
cv2.circle(im_bw, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
cv2.circle(im_bw, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
cv2.circle(im_bw, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
# draw lines between the midpoints
cv2.line(im_bw, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
         (255, 0, 255), 10)
cv2.line(im_bw, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
         (255, 0, 255), 10)

# compute the Euclidean distance between the midpoints
dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
dimA = dA * ConstPixelSpacing[0] * 0.1
dimB = dB * ConstPixelSpacing[1] * 0.1

# print("Altura: ", dA*0.065, "mm. Largura: ",  dB*0.065, "mm")
print("Altura: ", dimA, "mm. Largura: ", dimB, "mm")

# draw the object sizes on the image
cv2.putText(im_bw, "{:.1f}cm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            10, (255, 0, 0), 20)
cv2.putText(im_bw, "{:.1f}cm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            10, (255, 0, 0), 20)

pyplot.imshow(im_bw, cmap=pyplot.cm.bone), pyplot.title('Measurements')
pyplot.show()

# Volume calculation
breastThickness = RefDs.BodyPartThickness * 0.1  # cm
breastVolume = (math.pi / 4) * (dimA * dimB * breastThickness)
print("Breast volume: ", breastVolume)
