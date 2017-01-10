Simple Image Encryption using Hash and Random LSB Embedding<br/>
how to: <br/>

> python rgbvcs.py --help

This is my research for my bachelor degree.<br/>
**Slow computation and inefficient code** since i didn't have much time and i don't really understand about algorithm and data structure.<br/>
The encryption algorithm is so simple, explained below:

1. Get the hash value from the key image
2. Random pixel permutation on plain image using PRNG put hash value as seed for the PRNG.
3. Simple diffusion on plain image using xor.
4. Get the hash value from the image array of plain image.
5. Random pixel permutation on key image using PRNG and put hash value from the plain image as seed.
6. XOR the pixels from plain image and key image.
7. Using PSNR, randomly embed the hash value in to the ciphered image using LSB.
8. Save the image.
