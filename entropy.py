from imagecrypt import entropy
import cv2
import sys
img = cv2.imread(sys.argv[1])
print 'entropy :',sys.argv[1],":",entropy(img)
