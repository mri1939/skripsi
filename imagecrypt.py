from sys import stdout, stderr
import cv2
import numpy as np
import random
import math
import os
import hashlib
import binascii


class Image:

    def __init__(self):
        self.image = None
        self.height, self.width = (0, 0)
        self.name = None
        self.location = None
        self.pixel_count = 0

    def load_image(self, imageloc):
        self.name = os.path.basename(imageloc)
        self.location = imageloc
        try:
            self.image = cv2.imread(imageloc)
            self.height, self.width = self.image.shape[:2]
            self.pixel_count = self.height*self.width
        except AttributeError:
            stderr.write("\x1b[1;31mError: %s Not Found!\n\x1b[m" % self.name)
            exit()

    def set_size(self):
        self.height, self.width = self.image.shape[:2]

    def set_name(self, name):
        self.name = name

    def get_image(self):
        """
        :return: return the Image Array
        :rtype: Image
        """
        return self.image

    def set_image(self, image_arr):
        self.image = image_arr

    def permutate(self, keyhash):
        """
        :param keyhash: Hash value of the Key Image
        :type keyhash:str
        :return: none
        :rtype: none
        """
        imagearr = self.image.tolist()
        img = self.image
        random.seed(keyhash)
        progress = 0
        #np.random.permutation()
        total = self.width*self.height
        for i in xrange(0, self.height):
            for j in xrange(0, self.width):
                progress = (progress+1)
                if imagearr[i][j] is None:
                    continue
                while True:
                    r1 = random.randint(0, self.width-1)
                    r2 = random.randint(0, self.height-1)
                    if imagearr[r2][r1] is not None:
                        x = imagearr[r2][r1]
                        imagearr[r2][r1] = None
                        imagearr[i][j] = None
                        img[r2][r1] = img[i][j]
                        img[i][j] = np.array(x)
                        break
            stdout.write("\r\x1b[1;32mPermutation Process : %.2f %%" % (float(progress)/float(total)*100))
            stdout.flush()
        stdout.write("\n\x1b[1;37m")
        self.image = img

    def save_image(self, output):
        """
        Save Image to the given output filename.
        :param output: Output Filename
        :type output: str
        :return: None
        :rtype: None
        """
        cv2.imwrite(output, self.image)
        print "Image {} saved!".format(output)

    def show_image(self, window_name):
        """
        Show Image using using imshow from OpenCV
        :param window_name: Window Name
        :type window_name: str
        :return: none
        :rtype: None
        """
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_rgb_value(image):
    """
    :param image: cvimage
    :return: r, g, b
    :rtype: tupple
    """
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    return r, g, b


def sha1file(filepath):
    """
    :param filepath: Filename (/path/to/imagename)
    :type filepath : str
    :return: SHA1Sum of the File.
    :rtype: str
    """
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def sha1image(imagearr):
    """
    :param imagearr:image array
    :type imagearr:ndarray
    :return:sha1 string
    """
    return hashlib.sha1(imagearr).hexdigest()


def random_embed(image, seed, chksum):
    """
    LSB embedding using PRNG
    :param image: Image
    :param seed: Seed for the PRNG
    :param chksum: Chksum to be embedded
    :return: Embedded Image
    :rtype: Image Array
    """
    random.seed(seed)
    arr = np.resize(image.get_image(),(image.width*image.height*3))
    binseed = binascii.unhexlify(chksum)
    binseed = ''.join(format(ord(i),'b').zfill(8) for i in binseed)
    used = []
    for i in binseed:
        while True:
            rand = random.randint(0,len(arr))
            if rand not in used:
                used.append(rand)
                break
        if i == '0':
            arr[rand] &= 0b11111110
        else:
            arr[rand] |= 0b00000001
    arr = np.resize(arr,(image.width,image.height,3))
    return arr


def extract_chksum(image, seed, chksum_len):
    """
    LSB extraction to extract the chksum from the Encrypted Image
    :image: The encrypted image
    :seed: seed to be used for extracting the chksum
    :return: The hidden chksum
    :rtype: str
    """
    chksum=''
    random.seed(seed)
    arr = np.resize(image.get_image(), (image.width*image.height*3))
    used = []
    for i in xrange(chksum_len):
        while True:
            rand = random.randint(0, len(arr))
            if rand not in used:
                used.append(rand)
                break
        chksum += bin(arr[rand])[-1:]
    # print chksum
    bin_list = [chksum[i:i+8] for i in xrange(0, chksum_len, 8)]
    # print bin_list
    chksum = ''.join(binascii.hexlify(chr(int(x, 2))) for x in bin_list)
    return chksum


def md5sum(filepath):
    """
    :param filepath: Filename (/path/to/imagename)
    :type filepath : str
    :return: SHA1Sum of the File.
    :rtype: str
    """
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _diffusion(img):
    img_arr = img.image
    block_size = 1
    img_arr = np.reshape(img_arr, ((img.pixel_count/block_size), block_size, 3))
    count = 0
    for x in img_arr:
        if count != 0:
            img_arr[count] = np.bitwise_xor(x, img_arr[count-1])
        count += 1
        stdout.write("\r\x1b[1;32mDiffusion Process")
        stdout.flush()
    stdout.write("\n\x1b[1;37m")
    img_arr = np.reshape(img_arr, (img.width, img.height, 3))
    return img_arr


def _diffusion2(img):
    img_arr = img.image
    block_size = 1
    img_arr = np.reshape(img_arr, ((img.pixel_count/block_size), block_size, 3))
    count = (img.pixel_count/(block_size**2))-1
    for x in img_arr[::-1]:
        if count != 0:
            img_arr[count] = np.bitwise_xor(x, img_arr[count-1])
        count -= 1
        stdout.write("\r\x1b[1;32mReverse Diffusion Process")
        stdout.flush()
    stdout.write("\n\x1b[1;37m")
    img_arr = np.reshape(img_arr, (img.width, img.height, 3))
    return img_arr


def _xor(image, key):
    """
    Encryption Function
    :param image: Key Image
    :param key: Key Image
    :type image: Image
    :type key : Image
    :return: Encrypted Image
    :rtype : numpy_array
    """
    stdout.write("\r\x1b[1;32mXOR-ing Process")
    stdout.flush()
    stdout.write("\n\x1b[1;37m")
    res = image  # membuat gambar kosong
    # image = _diffusion(image)
    # key = _diffusion(key)
    res_image = np.bitwise_xor(image.image,key.image)
    res.set_image(res_image)
    res.set_size()
    res.set_name("result.png")
    return res


def encrypt(image, key_image, output_file):
    hash1 = sha1file(key_image.location)
    image.permutate(hash1)
    # key_image.permutate(hash1)
    image.set_image(_diffusion(image))
    chksum = sha1image(image.get_image())
    key_image.permutate(chksum)
    result = _xor(image, key_image)
    result.set_image(random_embed(result, hash1, chksum))
    result.save_image(output_file)


def decrypt(image, key_image, output_file):
    embedded_length = 160
    hash1 = sha1file(key_image.location)
    chksum = extract_chksum(image, hash1, embedded_length)
    key_image.permutate(chksum)
    result = _xor(image, key_image)
    result.set_image(_diffusion2(result))
    result.permutate(hash1)
    result.save_image(output_file)


def mse(imagename1, imagename2):
    """
    Calculate the Mean Sequence Error of 2 given Images
    :param imagename1: Image Filename
    :param imagename2: Image Filname
    :type imagename1: str
    :type imagename2: str
    :return: MSE Value
    :rtype: float
    """
    img = cv2.imread(imagename1)
    img2 = cv2.imread(imagename2)
    err = np.sum((img.astype("float") - img2.astype("float"))**2)
    err /= float(img.shape[0]*img.shape[1])
    return err


def psnr(imagename1, imagename2):
    """
    Calculate the PSNR of two images
    :param imagename1: Image Filename
    :param imagename2: Image Filname
    :type imagename1: str
    :type imagename2: str
    :return: MSE Value
    :rtype: float
    """
    mean_error = mse(imagename1, imagename2)
    res = 10*math.log10((255**2)/mean_error)
    return res


def entropy(color):
    arr = [0]
    arr = np.array(arr, dtype=np.float64)
    for i in np.arange(0, 256):
        nl = np.sum(color == i)
        # print nl
        if nl == 0:
            continue
        t = color.size
        pl = float(nl)/float(t)
        # print pl
        x = pl*np.log2(1/pl)
        arr = np.append(arr, x)
    return arr.sum()


def npcr(image1loc,image2loc):
    c1 = cv2.imread(image1loc)
    c2 = cv2.imread(image2loc)

    c1 = np.resize(c1, (512 ** 2, 3))
    c2 = np.resize(c2, (512 ** 2, 3))
    # jumlah = [0 if np.array_equal(x,y) else 1 for x in c1 for y in c2]
    jumlah = 0
    for x in xrange(0, 512 ** 2):
        # print x
        if not np.array_equal(c1[x], c2[x]): jumlah += 1
    return (jumlah / 512.0 ** 2) * 100


def uaci(imagename1, imagename2):
    c1 = cv2.imread(imagename1)
    c2 = cv2.imread(imagename2)
    diff = np.sum(abs(c1.astype("float") - c2.astype("float")))
    diff /= 255.0
    return (diff / (512 * 512 * 3)) * 100
