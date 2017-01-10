from optparse import OptionParser
from imagecrypt import Image, encrypt, decrypt
from sys import stderr

usage = "%prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-a", "--action", dest="action", metavar="ACTION",
                  help="encrypt/decrypt")
parser.add_option("-s", "--secret", dest="secretimage", metavar="FILENAME",
                  help="Plain Image")
parser.add_option("-k", "--key", dest="key", metavar="FILENAME",
                  help="Key Image")
parser.add_option("-o", "--output", dest="output", metavar="FILENAME",
                  help="Output File (Default : out.png)")

parser.set_default("output", "out.png")
(option, args) = parser.parse_args()
imagename = option.secretimage
keyname = option.key
out = option.output

if imagename is None or keyname is None:
    stderr.write("\x1b[1;31mError! Please specify the secret and key images!\n\n\x1b[m")
    parser.print_help()
    exit()

img = Image()
img.load_image(imagename)
key = Image()
key.load_image(keyname)
if option.action == 'encrypt':
    encrypt(img, key, out)
elif option.action == 'decrypt':
    decrypt(img, key, out)
else:
    print "{} : Action is not available! \n".format(option.action)
    parser.print_help()
    exit()
