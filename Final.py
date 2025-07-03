# Dependencies
from PIL import Image
import numpy as np
import cv2

'''
The first three functions are existing algorithms that I included for testing and comparison purposes.
The first three functions are not apart of my created algorithm. My algorithm to be evaluated is RGBHash.
Each of the four functions produce a hash value, hamming distance and similarity score between 2 images,
which are intended to be used for comparison.
'''


# Average Hash Algorithm (Existing algorithm)
def ahash(image_path, hash_size=8):
    # Load image, convert to grayscale, and resize
    image = Image.open(image_path).convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    pixels = np.array(image)

    # Compute average pixel value
    avg = pixels.mean()

    # Create hash: 1 if pixel > avg, else 0
    diff = pixels > avg

    # Convert binary array to hexadecimal hash
    hash_bits = ''.join('1' if bit else '0' for bit in diff.flatten())
    hex_hash = '{:0{}x}'.format(int(hash_bits, 2), hash_size * hash_size // 4)

    return hex_hash

# Perceptual Hash Algorithm (Existing algorithm)
def phash(image_path, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    image = Image.open(image_path).convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)

    # Convert to NumPy array and apply DCT (via OpenCV)
    pixels = np.array(image, dtype=np.float32)
    dct = cv2.dct(pixels)

    # Extract top-left DCT coefficients (excluding DC component)
    dct_low_freq = dct[:hash_size, :hash_size]
    avg = dct_low_freq.mean()
    diff = dct_low_freq > avg

    # Convert to hexadecimal hash
    hash_bits = ''.join('1' if bit else '0' for bit in diff.flatten())
    hex_hash = '{:0{}x}'.format(int(hash_bits, 2), hash_size * hash_size // 4)
    return hex_hash

# Difference Hash Algorithm (Existing Algorithm)
def dhash(image_path, hash_size=8):
    # Resize width to hash_size + 1 for horizontal difference
    image = Image.open(image_path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    # Compare adjacent columns
    diff = pixels[:, 1:] > pixels[:, :-1]

    # Convert to hexadecimal hash
    hash_bits = ''.join('1' if bit else '0' for bit in diff.flatten())
    hex_hash = '{:0{}x}'.format(int(hash_bits, 2), hash_size * hash_size // 4)
    return hex_hash


# My Algorithm: RGB Hash Algorithm
# This algorithm cross compares RGB channels, and performs average hash on grayscale and RGB channels
def RGBhash(image_path, hash_size=8):
    # opens image path and resizes to hash size
    image = (Image.open(image_path)
             .resize((hash_size, hash_size),
                     Image.Resampling.LANCZOS))

    # convert to grayscale and numpy array
    gray = image.convert('L')
    gray_pixels = np.array(gray)

    # split RGB channels and convert to numpy arrays
    r, g, b = image.split()
    r_pixels, g_pixels, b_pixels = np.array(r), np.array(g), np.array(b)

    # cross compare color channels to show which color dominates
    r_vs_g = r_pixels > g_pixels # true is red dominates green
    g_vs_b = g_pixels > b_pixels # true is green dominates blue
    b_vs_r = b_pixels > r_pixels # true is blue dominates red

    # creates hash from pixel average (aHash), compares each channel to its average
    gray_diff = gray_pixels > gray_pixels.mean()
    r_diff = r_pixels > r_pixels.mean()
    g_diff = g_pixels > g_pixels.mean()
    b_diff = b_pixels > b_pixels.mean()

    # helper function to convert bits to string
    def convertToHash(diff):
        return ''.join('1' if bit else '0' for bit in diff.flatten())

    # concatenates all hashes
    combinedHash = (convertToHash(gray_diff) + convertToHash(r_diff) +
                    convertToHash(g_diff) + convertToHash(b_diff) +
                    convertToHash(r_vs_g) + convertToHash(g_vs_b) +
                    convertToHash(b_vs_r))

    # converts to hexadecimal hash
    hex_hash = '{:0{}x}'.format(int(combinedHash, 2), len(combinedHash) // 4)
    return hex_hash

# computes hamming distance between 2 hashed images
def hamming_distance(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

# Mathematically computes a similarity score based on the hamming 
# distance and total bits for the two compared images
def similarityScore(hash1, hash2):
    distance = hamming_distance(hash1, hash2)
    totalBits = len(hash1) * 4
    similarity = (1-distance / totalBits)
    return similarity

# loads image variables
# image numbers may be changed to compare different images
img1 = 'image1.jpg'
img2 = 'image2.jpg'

# Average Hash 
hash_ah1 = ahash(img1)
hash_ah2 = ahash(img2)
print("aHash")
print("Hash1:", hash_ah1)
print("Hash2:", hash_ah2)
print("aHash Hamming Distance:", hamming_distance(hash_ah1, hash_ah2))
score = similarityScore(hash_ah1, hash_ah2)
print(f"Similarity Score: {score:.2%}")

# Perceptual Hash
hash_ph1 = phash(img1)
hash_ph2 = phash(img2)
print("pHash")
print("Hash1:", hash_ph1)
print("Hash2:", hash_ph2)
print("pHash Hamming Distance:", hamming_distance(hash_ph1, hash_ph2))
score = similarityScore(hash_ph1, hash_ph2)
print(f"Similarity Score: {score:.2%}")

# Difference Hash
hash_dh1 = dhash(img1)
hash_dh2 = dhash(img2)
print("dHash")
print("Hash1:", hash_dh1)
print("Hash2:", hash_dh2)
print("dHash Hamming Distance:", hamming_distance(hash_dh1, hash_dh2))
score = similarityScore(hash_dh1, hash_dh2)
print(f"Similarity Score: {score:.2%}")

# RGB Hash
hash_RGB1 = RGBhash(img1)
hash_RGB2 = RGBhash(img2)
print("RGBHash")
print("Hash1:", hash_RGB1)
print("Hash2:", hash_RGB2)
print("RGBHash Hamming Distance:", hamming_distance(hash_RGB1, hash_RGB2))
score = similarityScore(hash_RGB1, hash_RGB2)
print(f"Similarity Score: {score:.2%}")
