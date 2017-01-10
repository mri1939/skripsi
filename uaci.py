import sys
from imagecrypt import uaci,npcr

img1 = sys.argv[1]
img2 = sys.argv[2]
print "UACI : ",uaci(img1,img2)
print "NPCR : ",npcr(img1,img2)
