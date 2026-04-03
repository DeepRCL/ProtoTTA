"""
Corruption utilities for robustness testing, adapted from ImageNet-C.
13 corruption types used in ProtoTTA paper.

  Noise:   gaussian_noise, shot_noise, impulse_noise, speckle_noise
  Blur:    gaussian_blur, defocus_blur
  Weather: fog, frost, brightness
  Digital: jpeg_compression, contrast, pixelate, elastic_transform
"""
import warnings
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    from io import BytesIO
    from scipy.ndimage import zoom as scizoom
    from scipy.ndimage.interpolation import map_coordinates
    import skimage as sk
    from skimage.filters import gaussian
    import cv2
    CORRUPTIONS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Some corruption deps not available: {e}. Only basic corruptions will work.")
    CORRUPTIONS_AVAILABLE = False

warnings.simplefilter("ignore", UserWarning)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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


def plasma_fractal(mapsize=256, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

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


# ---------------------------------------------------------------------------
# The 13 required corruption functions
# ---------------------------------------------------------------------------

# --- Noise ---

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# --- Blur ---

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


# --- Weather ---

def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    h, w = np.array(x).shape[:2]
    x = np.array(x) / 255.
    max_val = x.max()
    mapsize = max(256, 2 ** int(np.ceil(np.log2(max(h, w)))))
    plasma = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:h, :w]
    x += c[0] * plasma[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    """Frost corruption — falls back to additive noise if frost images unavailable."""
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    # Fallback: simple noise pattern (avoids hard dependency on frost image files)
    x_np = np.array(x, dtype=np.float32)
    rng = np.random.RandomState(42)
    noise = rng.randint(0, 256, size=x_np.shape, dtype=np.uint8).astype(np.float32)
    return np.clip(c[0] * x_np + c[1] * noise, 0, 255)


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


# --- Digital ---

def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    return x


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    w, h = x.size
    x = x.resize((int(w * c), int(h * c)), Image.BOX)
    x = x.resize((w, h), Image.BOX)
    return x


def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (np.reshape(y + dy, (-1, 1)),
               np.reshape(x + dx, (-1, 1)),
               np.reshape(z, (-1, 1)))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# ---------------------------------------------------------------------------
# Registry  (exactly the 13 corruptions we want)
# ---------------------------------------------------------------------------

CORRUPTION_DICT = {
    # Noise
    'gaussian_noise':  gaussian_noise,
    'shot_noise':      shot_noise,
    'impulse_noise':   impulse_noise,
    'speckle_noise':   speckle_noise,
    # Blur
    'gaussian_blur':   gaussian_blur,
    'defocus_blur':    defocus_blur,
    # Weather
    'fog':             fog,
    'frost':           frost,
    'brightness':      brightness,
    # Digital
    'jpeg_compression': jpeg_compression,
    'contrast':        contrast,
    'pixelate':        pixelate,
    'elastic_transform': elastic_transform,
}

CORRUPTION_TYPES = list(CORRUPTION_DICT.keys())


def get_all_corruption_types():
    return CORRUPTION_TYPES


# ---------------------------------------------------------------------------
# PyTorch transform wrapper
# ---------------------------------------------------------------------------

class AddCorruptions:
    """Callable that applies a corruption inside a transforms.Compose pipeline."""

    def __init__(self, corruption_type, severity=1):
        if corruption_type not in CORRUPTION_DICT:
            raise ValueError(f"Unknown corruption: {corruption_type}. "
                             f"Available: {CORRUPTION_TYPES}")
        self.fn = CORRUPTION_DICT[corruption_type]
        self.corruption_type = corruption_type
        self.severity = severity

    def __call__(self, img):
        """img is a PIL Image → returns PIL Image (float-safe)."""
        try:
            result = self.fn(img, self.severity)
            if isinstance(result, np.ndarray):
                result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
            elif not isinstance(result, Image.Image):
                result = img  # fallback
            return result
        except Exception as e:
            warnings.warn(f"Corruption {self.corruption_type} failed: {e}. Using clean image.")
            return img


def get_corrupted_transform(img_size, mean, std, corruption_type=None, severity=1):
    """Return a transform pipeline that optionally includes a corruption.

    Returns a Compose that yields *normalised tensors*.
    """
    t = [transforms.Resize(size=(img_size, img_size))]

    if corruption_type and corruption_type in CORRUPTION_DICT:
        # Corruption is applied on PIL images before ToTensor
        t.append(AddCorruptions(corruption_type, severity))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)
