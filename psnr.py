import sys
from imagecrypt import mse,psnr
img1 = sys.argv[1]
img2 = sys.argv[2]
print "MSE : ",mse(img1,img2)
print "PSNR : ",psnr(img1,img2)
