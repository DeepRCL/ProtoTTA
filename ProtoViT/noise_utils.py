"""
Corruption utilities for robustness testing, adapted from ImageNet-C.
Supports all 19 corruption types from the ImageNet-C benchmark.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings

# Import corruption functions from ImageNet-C
try:
    from io import BytesIO
    from scipy.ndimage import zoom as scizoom
    from scipy.ndimage.interpolation import map_coordinates
    import skimage as sk
    from skimage.filters import gaussian
    import cv2
    from wand.image import Image as WandImage
    from wand.api import library as wandlibrary
    import ctypes
    CORRUPTIONS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Some corruption dependencies not available: {e}. Only basic corruptions will work.")
    CORRUPTIONS_AVAILABLE = False

warnings.simplefilter("ignore", UserWarning)

# ============================================================================
# Corruption Helper Functions (from ImageNet-C)
# ============================================================================

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


if CORRUPTIONS_AVAILABLE:
    wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)
    
    class MotionImage(WandImage):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# ============================================================================
# Corruption Functions (adapted from ImageNet-C for any image size)
# ============================================================================

def gaussian_noise(x, severity=1):
    """Add Gaussian noise."""
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    """Add shot (Poisson) noise."""
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    """Add impulse (salt-and-pepper) noise."""
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    """Add speckle noise."""
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    """Apply Gaussian blur."""
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    """Apply glass blur effect."""
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    
    # Get image size dynamically
    h, w = x.shape[:2]
    
    # Locally shuffle pixels
    for i in range(c[2]):
        for hh in range(h - c[1], c[1], -1):
            for ww in range(w - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = hh + dy, ww + dx
                x[hh, ww], x[h_prime, w_prime] = x[h_prime, w_prime], x[hh, ww]
    
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    """Apply defocus blur."""
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    
    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    """Apply motion blur."""
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    
    output = BytesIO()
    x.save(output, format='PNG')
    x_wand = MotionImage(blob=output.getvalue())
    x_wand.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    
    x_blur = cv2.imdecode(np.frombuffer(x_wand.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    if x_blur.shape != x.size[::-1]:
        return np.clip(x_blur[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x_blur, x_blur, x_blur]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    """Apply zoom blur."""
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    """Add fog effect."""
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    
    h, w = np.array(x).shape[:2]
    x = np.array(x) / 255.
    max_val = x.max()
    
    # Generate plasma fractal at the appropriate size
    mapsize = max(256, 2 ** int(np.ceil(np.log2(max(h, w)))))
    plasma = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:h, :w]
    
    x += c[0] * plasma[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, frost_path='./robustness/ImageNet-C/imagenet_c/imagenet_c/frost/'):
    """Add frost effect."""
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    
    import os
    frost_files = [
        os.path.join(frost_path, f'frost{i}.png' if i <= 3 else f'frost{i}.jpg')
        for i in range(1, 7)
    ]
    
    # Check if frost images exist
    if not os.path.exists(frost_files[idx]):
        # Fallback: use a simple noise pattern
        warnings.warn(f"Frost image not found at {frost_files[idx]}, using noise fallback")
        noise = np.random.randint(0, 256, size=np.array(x).shape, dtype=np.uint8)
        return np.clip(c[0] * np.array(x) + c[1] * noise, 0, 255)
    
    frost_img = cv2.imread(frost_files[idx])
    h, w = np.array(x).shape[:2]
    
    # Randomly crop
    x_start = np.random.randint(0, max(1, frost_img.shape[0] - h))
    y_start = np.random.randint(0, max(1, frost_img.shape[1] - w))
    frost_crop = frost_img[x_start:x_start + h, y_start:y_start + w][..., [2, 1, 0]]
    
    return np.clip(c[0] * np.array(x) + c[1] * frost_crop, 0, 255)


def snow(x, severity=1):
    """Add snow effect."""
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
    
    x_np = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x_np.shape[:2], loc=c[0], scale=c[1])
    
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    
    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())
    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    
    x_gray = cv2.cvtColor(x_np.astype(np.float32), cv2.COLOR_RGB2GRAY).reshape(x_np.shape[0], x_np.shape[1], 1)
    x_np = c[6] * x_np + (1 - c[6]) * np.maximum(x_np, x_gray * 1.5 + 0.5)
    return np.clip(x_np + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    """Add spatter effect."""
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x_np = np.array(x, dtype=np.float32) / 255.
    
    liquid_layer = np.random.normal(size=x_np.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        
        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x_np = cv2.cvtColor(x_np, cv2.COLOR_BGR2BGRA)
        
        return cv2.cvtColor(np.clip(x_np + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        
        color = np.concatenate((63 / 255. * np.ones_like(x_np[..., :1]),
                                42 / 255. * np.ones_like(x_np[..., :1]),
                                20 / 255. * np.ones_like(x_np[..., :1])), axis=2)
        
        color *= m[..., np.newaxis]
        x_np *= (1 - m[..., np.newaxis])
        
        return np.clip(x_np + color, 0, 1) * 255


def contrast(x, severity=1):
    """Reduce contrast."""
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    """Increase brightness."""
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    """Adjust saturation."""
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    """Apply JPEG compression."""
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    return x


def pixelate(x, severity=1):
    """Apply pixelation."""
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    w, h = x.size
    x = x.resize((int(w * c), int(h * c)), Image.BOX)
    x = x.resize((w, h), Image.BOX)
    return x


def elastic_transform(image, severity=1):
    """Apply elastic transformation."""
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# ============================================================================
# Corruption Registry
# ============================================================================

CORRUPTION_DICT = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'speckle_noise': speckle_noise,
    'gaussian_blur': gaussian_blur,
    'glass_blur': glass_blur,
    'defocus_blur': defocus_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'fog': fog,
    'frost': frost,
    'snow': snow,
    # 'spatter': spatter,
    'contrast': contrast,
    'brightness': brightness,
    # 'saturate': saturate,
    'jpeg_compression': jpeg_compression,
    'pixelate': pixelate,
    'elastic_transform': elastic_transform,
}


# ============================================================================
# PyTorch Transform Wrapper
# ============================================================================

class AddCorruptions:
    """Wrapper to add various corruptions to images in a PyTorch pipeline."""
    
    def __init__(self, corruption_type='gaussian_noise', severity=1):
        """
        Args:
            corruption_type: Type of corruption (see CORRUPTION_DICT keys)
            severity: Severity level (1-5, where 5 is most severe)
        """
        self.corruption_type = corruption_type
        self.severity = severity
        
        if corruption_type not in CORRUPTION_DICT:
            raise ValueError(f"Unknown corruption type: {corruption_type}. "
                           f"Available: {list(CORRUPTION_DICT.keys())}")
        
        self.corruption_fn = CORRUPTION_DICT[corruption_type]
        
    def __call__(self, img):
        """
        Apply corruption to an image.
        
        Args:
            img: PIL Image
            
        Returns:
            Tensor with corruption applied
        """
        # Apply corruption (expects PIL Image)
        try:
            corrupted = self.corruption_fn(img, self.severity)
            
            # Handle different return types
            if isinstance(corrupted, np.ndarray):
                corrupted = Image.fromarray(corrupted.astype(np.uint8))
            
            # Convert to tensor
            return transforms.ToTensor()(corrupted)
        except Exception as e:
            warnings.warn(f"Corruption {self.corruption_type} failed: {e}. Returning uncorrupted image.")
            return transforms.ToTensor()(img)


def get_corrupted_transform(img_size, mean, std, corruption_type=None, severity=1):
    """
    Returns a transform pipeline that includes corruption if specified.
    
    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        corruption_type: Type of corruption (None for clean data)
        severity: Severity level (1-5)
    
    Returns:
        Composed transform pipeline
    """
    base_transforms = [
        transforms.Resize(size=(img_size, img_size)),
    ]
    
    if corruption_type and corruption_type in CORRUPTION_DICT:
        # Add corruption directly (it returns a tensor)
        base_transforms.append(AddCorruptions(corruption_type, severity))
    else:
        # No corruption, just convert to tensor
        base_transforms.append(transforms.ToTensor())
        
    # Normalization must happen LAST, after noise addition
    base_transforms.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(base_transforms)


def get_all_corruption_types():
    """Returns list of all available corruption types."""
    return list(CORRUPTION_DICT.keys())

