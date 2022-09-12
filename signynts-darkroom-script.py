import numpy as np
import cv2
from skimage import color
import os
from os import listdir
from skimage import img_as_float
from skimage import exposure
import scipy.ndimage

# global variables
cutoff_margin = 10
blur_size = (7, 7)
QuantumRange = 65535
GammaGlobal = 2.15

# load image

def normalize(image, low, high):
    float = img_as_float(image)
    p_low, p_high = np.percentile(float, (low, high))
    rescale = exposure.rescale_intensity(float, in_range=(p_low, p_high))
    normalized = rescale * QuantumRange/rescale.max()
    return normalized

def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = ((np.arange(0, (QuantumRange + 1)) / QuantumRange) ** inv_gamma) * QuantumRange
    table = table.astype(np.uint16)
    return table[image.astype(np.uint16)]

def adjust_channel(channel, mininum, gamma):
    normalized = channel / QuantumRange
    exponent = np.clip((mininum * normalized.astype(float) ** (-1)), 0, QuantumRange)
    output = exponent * QuantumRange
    clipped = np.clip(output, 0, QuantumRange)
    adjusted_channel = adjust_gamma(clipped, gamma)
    return adjusted_channel

def recompile_image(r_channel, g_channel, b_channel, gamma_subtract):
    image = (cv2.merge([r_channel.astype(np.uint16), g_channel.astype(np.uint16), b_channel.astype(np.uint16)]))
    image = adjust_gamma(image.astype(np.uint16), GammaGlobal)
    image = image.astype(np.uint16) - gamma_subtract
    image = np.clip(image, np.amin(image), np.amax(image))
    recompiled_image = cv2.normalize(image, None, alpha=0, beta=QuantumRange, norm_type=cv2.NORM_MINMAX)
    return recompiled_image

def autolevel_image(image):
    image_gray = color.rgb2gray(image.astype(np.uint16))

    blackpoint = np.amin(image_gray) * QuantumRange
    whitepoint = np.amax(image_gray) * QuantumRange
    autolevel_mean = 100 * np.mean(image) / QuantumRange
    autolevel_midrange = 0.5
    autolevel_gamma = np.log(autolevel_mean / 100) / np.log(autolevel_midrange)

    image_hsv = color.rgb2hsv(image.astype(np.uint16))
    h_channel,s_channel,v_channel = cv2.split(image_hsv)

    v_channel = v_channel * QuantumRange
    v_channel = np.clip(v_channel, blackpoint, whitepoint)
    v_channel = cv2.normalize(v_channel, None, alpha=0, beta=QuantumRange, norm_type=cv2.NORM_MINMAX)
    v_channel = v_channel / QuantumRange

    output_hsv = cv2.merge((h_channel, s_channel, v_channel))
    autoleveled_image = color.hsv2rgb(output_hsv) * QuantumRange
    autoleveled_image = adjust_gamma(autoleveled_image.astype(np.uint16), autolevel_gamma)

    return autoleveled_image

def autocolor_image(image):
    image_gray = color.rgb2gray(image.astype(np.uint16))
    r_channel, g_channel, b_channel = cv2.split(image)

    neutral_gray = np.mean(image_gray)

    r_mean = (np.mean(r_channel) / QuantumRange)
    r_ratio = neutral_gray / r_mean
    g_mean = (np.mean(g_channel) / QuantumRange)
    g_ratio = neutral_gray / g_mean
    b_mean = (np.mean(b_channel) / QuantumRange)
    b_ratio = neutral_gray / b_mean

    r_colorcorrected = np.clip((r_channel * b_ratio), 0, QuantumRange) # prevents clipping
    g_colorcorrected = np.clip((g_channel * g_ratio), 0, QuantumRange)
    b_colorcorrected = np.clip((b_channel * r_ratio), 0, QuantumRange)

    autocolored_image = (cv2.merge([r_colorcorrected.astype(np.uint16), g_colorcorrected.astype(np.uint16), b_colorcorrected.astype(np.uint16)]))

    return autocolored_image
# process image
 
def negative_inversion(image_input):
    r_input, g_input, b_input = cv2.split(image_input)

    # def find_gamma(image_input, cutoff_margin):
    image_crop = image_input[cutoff_margin:-cutoff_margin, cutoff_margin:-cutoff_margin]
    image_blur = cv2.blur(image_crop, blur_size)
    r_blur, g_blur, b_blur = cv2.split(image_blur)

    r_min = np.amin(r_blur) / QuantumRange
    r_max = np.amax(r_blur) / QuantumRange
    g_min = np.amin(g_blur) / QuantumRange
    g_max = np.amax(g_blur) / QuantumRange
    b_min = np.amin(b_blur) / QuantumRange
    b_max = np.amax(b_blur) / QuantumRange

    gamma_subtract = ((b_min/b_max) ** (1/GammaGlobal)) * QuantumRange * 0.95
    gamma_for_r = np.log(r_max/r_min) / np.log(b_max/b_min)
    gamma_for_g = np.log(g_max/g_min) / np.log(b_max/b_min)
    gamma_for_b = 1

    r_adjusted = adjust_channel(r_input, r_min, gamma_for_r)
    g_adjusted = adjust_channel(g_input, g_min, gamma_for_g)
    b_adjusted = adjust_channel(b_input, b_min, gamma_for_b)

    inverted_negative = recompile_image(r_adjusted, g_adjusted, b_adjusted, gamma_subtract)
    autoleveled_negative = autolevel_image(inverted_negative)
    autocolored_negative = autocolor_image(autoleveled_negative)

    return autocolored_negative, r_input

def remove_noise(image, aggressiveness=7):
    normalized = image/QuantumRange
    inverted = 1-normalized
    tmp = scipy.ndimage.convolve(inverted, np.ones((3,3)), mode='constant')
    out = np.logical_and(tmp >= aggressiveness, inverted).astype(np.float32)
    reverted = 1-out
    unnormalized = reverted * QuantumRange
    return unnormalized

def dust_removal(infrared_image, inverted_image, r_channel):
    normalized_infrared = normalize(infrared_image, 2, 99)
    normalized_red = normalize(r_channel, 2, 99)

    c = normalized_infrared/((normalized_red.astype('float')+1)/(QuantumRange+1))
    divided = c*(c < QuantumRange)+QuantumRange*np.ones(np.shape(c))*(c > QuantumRange)

    ret, threshold = cv2.threshold(divided,(QuantumRange-QuantumRange/1),QuantumRange,cv2.THRESH_BINARY)
    
    pass1 = remove_noise(threshold)
    pass2 = remove_noise(pass1)
    pass3 = remove_noise(pass2)

    return pass3

def signynts_darkroom_script(file_input):
    layered, negative_input = cv2.imreadmulti(file_input, [], cv2.IMREAD_UNCHANGED) # it returns if the image is multilayered (layered)
    inverted_image, r_channel = negative_inversion(negative_input[0])
    dust_removed = dust_removal(negative_input[1], inverted_image, r_channel)
    inverted_image = dust_removed

    return inverted_image

# actually process files

in_dir = 'input'
out_dir = 'output'

os.makedirs(out_dir, exist_ok=True)

filenames = []

for file in os.listdir(in_dir):
    if file.endswith(".tif"):
        filenames.append(file)

for file in filenames: 
    print(file)
    filepath = in_dir + '/' + file
    image = signynts_darkroom_script(filepath)
    cv2.imwrite(out_dir + '/' + file, image.astype(np.uint16), params=(cv2.IMWRITE_TIFF_COMPRESSION, 5))
