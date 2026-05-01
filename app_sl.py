import re
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, minimum_filter, maximum_filter
from scipy.signal import fftconvolve
from skimage import data
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
    richardson_lucy,
    unsupervised_wiener,
    wiener,
)

DEFAULT_IMAGE = data.coffee()

# Pre-compute initial slider bounds from the default image (coffee, 400 × 600)
_DEFAULT_H = DEFAULT_IMAGE.shape[0]   # 400
_DEFAULT_W = DEFAULT_IMAGE.shape[1]   # 600
_DEFAULT_MAX_RADIUS  = int(np.ceil(np.sqrt((_DEFAULT_H / 2.0) ** 2 + (_DEFAULT_W / 2.0) ** 2)))
_DEFAULT_MAX_KERNEL  = min(_DEFAULT_H, _DEFAULT_W) - (1 - min(_DEFAULT_H, _DEFAULT_W) % 2)
_DEFAULT_MAX_MOTION  = max(1, min(_DEFAULT_H, _DEFAULT_W))
_DEFAULT_MAX_KUWAHARA = max(1, min(_DEFAULT_H, _DEFAULT_W) // 8)
_DEFAULT_MAX_NOTCH_U = max(1, _DEFAULT_W // 2)
_DEFAULT_MAX_NOTCH_V = max(1, _DEFAULT_H // 2)
_DEFAULT_MAX_NOTCH_R = max(1, min(_DEFAULT_H, _DEFAULT_W) // 2)

LINEAR_FILTERS = [
    "Low-pass",
    "High-pass",
    "Ideal band-pass",
    "Ideal band-stop",
    "Gaussian low-pass",
    "Gaussian high-pass",
    "Butterworth low-pass",
    "Butterworth high-pass",
    "Notch reject",
]

NONLINEAR_FILTERS = [
    "Median filter",
    "Minimum filter",
    "Maximum filter",
    "Bilateral filter",
]

NOISE_REMOVAL_FILTERS = [
    "Median filter",
    "Gaussian filter",
    "Total variation",
    "Bilateral filter",
    "Kuwahara filter",
    "Non-local means",
    "Wavelet denoise",
]

DEBLUR_METHODS = [
    "Wiener",
    "Richardson-Lucy",
    "Unsupervised Wiener",
]

BLUR_TYPES = [
    "Gaussian blur",
    "Motion blur",
]

PHASE_MODIFICATION_METHODS = [
    "Phase-only reconstruction",
    "Magnitude-only reconstruction",
    "Random phase reconstruction",
    "Phase quantization",
    "Phase noise",
    "Linear phase ramp",
]

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

def load_markdown_file(filename: str) -> str:
    """Load a Markdown file next to this script, with a safe fallback for Streamlit Cloud/HF Spaces."""
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [base_dir / filename, Path.cwd() / filename]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


DOCUMENTATION_fr = load_markdown_file("documentation_fr.md")
DOCUMENTATION_en = load_markdown_file("documentation_en.md")


def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        title = lines[0].strip()

        if title.lower() in {"table des matières", "table of contents"}:
            continue

        sections[title] = "## " + part

    return sections


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr) if DOCUMENTATION_fr else {}
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en) if DOCUMENTATION_en else {}

DOC_FR_TITLES = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES = list(DOC_EN_SECTIONS.keys())


def load_doc_fr_section(title: str) -> str:
    return DOC_FR_SECTIONS[title]


def load_doc_en_section(title: str) -> str:
    return DOC_EN_SECTIONS[title]


def to_float_rgb(image):
    if image is None:
        return None
    arr = np.array(image).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(image):
    image = np.clip(image, 0.0, 1.0)
    return (255.0 * image).astype(np.uint8)


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def normalize_rgb_channels(image_rgb):
    out = np.zeros_like(image_rgb, dtype=np.float32)
    for c in range(image_rgb.shape[2]):
        out[:, :, c] = normalize01(image_rgb[:, :, c])
    return np.clip(out, 0.0, 1.0)


def rgb_to_gray(image_rgb):
    return (
        0.299 * image_rgb[:, :, 0]
        + 0.587 * image_rgb[:, :, 1]
        + 0.114 * image_rgb[:, :, 2]
    ).astype(np.float32)


def gray_to_rgb(gray):
    return np.stack([gray, gray, gray], axis=-1)


def odd_cap(value):
    value = max(1, int(value))
    if value % 2 == 0:
        value -= 1
    return max(1, value)


def map_phase_to_display(phase):
    return ((np.angle(np.exp(1j * phase)) + np.pi) / (2.0 * np.pi)).astype(np.float32)


def reconstruct_from_complex_spectrum(spectrum):
    spatial = np.fft.ifft2(spectrum)
    imag_energy = float(np.mean(np.abs(np.imag(spatial))))
    real_energy = float(np.mean(np.abs(np.real(spatial)))) + 1e-12
    if imag_energy < 1e-6 * real_energy:
        return np.real(spatial).astype(np.float32)
    return np.abs(spatial).astype(np.float32)


def get_image_geometry(image):
    image_rgb = to_float_rgb(image if image is not None else DEFAULT_IMAGE)
    h, w, _ = image_rgb.shape
    max_radius = int(np.ceil(np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)))
    max_kernel = odd_cap(min(h, w))
    max_motion = max(1, min(h, w))
    max_kuwahara = max(1, min(h, w) // 8)
    return h, w, max_radius, max_kernel, max_motion, max_kuwahara





def frequency_grid(h, w):
    y = np.arange(h) - h // 2
    x = np.arange(w) - w // 2
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    return xx, yy, d


def compute_fft_displays(image):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")

    gray = rgb_to_gray(image_rgb)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))

    magnitude = normalize01(np.log1p(np.abs(f_shift)))
    phase = map_phase_to_display(np.angle(f_shift))

    return to_uint8(gray_to_rgb(magnitude)), to_uint8(gray_to_rgb(phase))


def low_pass_mask(h, w, cutoff):
    _, _, d = frequency_grid(h, w)
    return (d <= cutoff).astype(np.float32)


def high_pass_mask(h, w, cutoff):
    return 1.0 - low_pass_mask(h, w, cutoff)


def band_pass_mask(h, w, cutoff_low, cutoff_high):
    _, _, d = frequency_grid(h, w)
    low = min(cutoff_low, cutoff_high)
    high = max(cutoff_low, cutoff_high)
    return ((d >= low) & (d <= high)).astype(np.float32)


def band_stop_mask(h, w, cutoff_low, cutoff_high):
    return 1.0 - band_pass_mask(h, w, cutoff_low, cutoff_high)


def gaussian_low_pass_mask(h, w, sigma):
    _, _, d = frequency_grid(h, w)
    sigma = max(float(sigma), 1e-6)
    return np.exp(-(d**2) / (2.0 * sigma**2)).astype(np.float32)


def gaussian_high_pass_mask(h, w, sigma):
    return 1.0 - gaussian_low_pass_mask(h, w, sigma)


def butterworth_low_pass_mask(h, w, cutoff, order):
    _, _, d = frequency_grid(h, w)
    cutoff = max(float(cutoff), 1e-6)
    order = max(int(order), 1)
    return (1.0 / (1.0 + (d / cutoff) ** (2 * order))).astype(np.float32)


def butterworth_high_pass_mask(h, w, cutoff, order):
    return 1.0 - butterworth_low_pass_mask(h, w, cutoff, order)


def notch_reject_mask(h, w, u0, v0, radius):
    xx, yy, _ = frequency_grid(h, w)
    d1 = np.sqrt((xx - u0) ** 2 + (yy - v0) ** 2)
    d2 = np.sqrt((xx + u0) ** 2 + (yy + v0) ** 2)
    return ((d1 > radius) & (d2 > radius)).astype(np.float32)


def build_linear_mask(filter_name, h, w, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius):
    if filter_name == "Low-pass":
        return low_pass_mask(h, w, cutoff1)
    if filter_name == "High-pass":
        return high_pass_mask(h, w, cutoff1)
    if filter_name == "Ideal band-pass":
        return band_pass_mask(h, w, cutoff1, cutoff2)
    if filter_name == "Ideal band-stop":
        return band_stop_mask(h, w, cutoff1, cutoff2)
    if filter_name == "Gaussian low-pass":
        return gaussian_low_pass_mask(h, w, cutoff1)
    if filter_name == "Gaussian high-pass":
        return gaussian_high_pass_mask(h, w, cutoff1)
    if filter_name == "Butterworth low-pass":
        return butterworth_low_pass_mask(h, w, cutoff1, order)
    if filter_name == "Butterworth high-pass":
        return butterworth_high_pass_mask(h, w, cutoff1, order)
    if filter_name == "Notch reject":
        return notch_reject_mask(h, w, notch_u, notch_v, notch_radius)
    return np.ones((h, w), dtype=np.float32)


def apply_frequency_mask_rgb(image_rgb, mask_centered):
    output = np.zeros_like(image_rgb, dtype=np.float32)
    for channel in range(3):
        f = np.fft.fft2(image_rgb[:, :, channel])
        f_shift = np.fft.fftshift(f)
        g_shift = f_shift * mask_centered
        g = np.fft.ifft2(np.fft.ifftshift(g_shift))
        output[:, :, channel] = np.real(g)
    return np.clip(output, 0.0, 1.0)


def integral_image(arr):
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def rect_sum(ii, top, left, bottom, right):
    return ii[bottom + 1, right + 1] - ii[top, right + 1] - ii[bottom + 1, left] + ii[top, left]


def kuwahara_filter_color(image_rgb, radius):
    radius = max(1, int(radius))
    h, w, _ = image_rgb.shape
    guide = rgb_to_gray(image_rgb)

    guide_pad = np.pad(guide, radius, mode="reflect")
    channel_pads = [np.pad(image_rgb[:, :, c], radius, mode="reflect") for c in range(3)]

    ii_g = integral_image(guide_pad)
    ii_g2 = integral_image(guide_pad**2)
    ii_channels = [integral_image(ch) for ch in channel_pads]

    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    area = float((radius + 1) ** 2)

    quadrants = [
        (y, x, y + radius, x + radius),
        (y, x + radius, y + radius, x + 2 * radius),
        (y + radius, x, y + 2 * radius, x + radius),
        (y + radius, x + radius, y + 2 * radius, x + 2 * radius),
    ]

    variances = []
    color_means = []
    for top, left, bottom, right in quadrants:
        sum_g = rect_sum(ii_g, top, left, bottom, right)
        sum_g2 = rect_sum(ii_g2, top, left, bottom, right)
        mean_g = sum_g / area
        var_g = np.maximum(sum_g2 / area - mean_g**2, 0.0)
        variances.append(var_g)

        means_rgb = []
        for ii_c in ii_channels:
            means_rgb.append(rect_sum(ii_c, top, left, bottom, right) / area)
        color_means.append(np.stack(means_rgb, axis=-1))

    variances = np.stack(variances, axis=-1)
    choice = np.argmin(variances, axis=-1)

    output = np.zeros_like(image_rgb)
    for idx in range(4):
        mask = choice == idx
        output[mask] = color_means[idx][mask]
    return np.clip(output, 0.0, 1.0)


def apply_nonlinear_filter(image_rgb, filter_name, kernel_size, sigma_spatial, sigma_color):
    kernel_size = odd_cap(kernel_size)

    if filter_name == "Median filter":
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = median_filter(image_rgb[:, :, channel], size=kernel_size)
        return np.clip(output, 0.0, 1.0)

    if filter_name == "Minimum filter":
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = minimum_filter(image_rgb[:, :, channel], size=kernel_size)
        return np.clip(output, 0.0, 1.0)

    if filter_name == "Maximum filter":
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = maximum_filter(image_rgb[:, :, channel], size=kernel_size)
        return np.clip(output, 0.0, 1.0)

    if filter_name == "Bilateral filter":
        output = denoise_bilateral(
            image_rgb,
            sigma_spatial=max(0.1, float(sigma_spatial)),
            sigma_color=max(0.01, float(sigma_color)),
            channel_axis=-1,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    return image_rgb


def get_filtered_image_rgb(image, filter_family, filter_name,
                           cutoff1, cutoff2, order, notch_u, notch_v, notch_radius,
                           kernel_size, sigma_spatial, sigma_color):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")

    if filter_family == "Linear" and filter_name in LINEAR_FILTERS:
        h, w, _ = image_rgb.shape
        mask = build_linear_mask(filter_name, h, w, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius)
        return apply_frequency_mask_rgb(image_rgb, mask)

    if filter_family == "Non-linear" and filter_name in NONLINEAR_FILTERS:
        return apply_nonlinear_filter(image_rgb, filter_name, kernel_size, sigma_spatial, sigma_color)

    raise ValueError("Please select a valid filter family and filter.")




def update_live_mask(image, filter_family, filter_name, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    h, w, _ = image_rgb.shape

    if filter_family == "Linear" and filter_name in LINEAR_FILTERS:
        mask = build_linear_mask(filter_name, h, w, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius)
        return to_uint8(gray_to_rgb(mask))

    return np.zeros((h, w, 3), dtype=np.uint8)


def apply_selected_filter(image, filter_family, filter_name,
                          cutoff1, cutoff2, order, notch_u, notch_v, notch_radius,
                          kernel_size, sigma_spatial, sigma_color):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")

    if filter_family == "Linear" and filter_name in LINEAR_FILTERS:
        h, w, _ = image_rgb.shape
        mask = build_linear_mask(filter_name, h, w, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius)
        result = apply_frequency_mask_rgb(image_rgb, mask)
        mask_rgb = gray_to_rgb(mask)
    elif filter_family == "Non-linear" and filter_name in NONLINEAR_FILTERS:
        result = apply_nonlinear_filter(image_rgb, filter_name, kernel_size, sigma_spatial, sigma_color)
        h, w, _ = image_rgb.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.float32)
    else:
        raise ValueError("Please select a valid filter family and filter.")

    result_gray = rgb_to_gray(result)
    result_fft = np.fft.fftshift(np.fft.fft2(result_gray))
    result_spectrum = normalize01(np.log1p(np.abs(result_fft)))
    return to_uint8(mask_rgb), to_uint8(gray_to_rgb(result_spectrum)), to_uint8(result)



def _build_modified_spectrum(
    spectrum,
    method,
    quant_levels,
    phase_noise_std,
    shift_x,
    shift_y,
    seed,
    shared_random_phase=None,
    shared_phase_noise=None,
):
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    h, w = spectrum.shape

    if method == "Phase-only reconstruction":
        modified_magnitude = np.ones_like(magnitude)
        modified_phase = phase

    elif method == "Magnitude-only reconstruction":
        modified_magnitude = magnitude
        modified_phase = np.zeros_like(phase)

    elif method == "Random phase reconstruction":
        modified_magnitude = magnitude
        if shared_random_phase is None:
            rng = np.random.default_rng(int(seed))
            shared_random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
        modified_phase = shared_random_phase

    elif method == "Phase quantization":
        modified_magnitude = magnitude
        levels = max(2, int(quant_levels))
        step = 2.0 * np.pi / float(levels)
        modified_phase = -np.pi + step * np.round((phase + np.pi) / step)
        modified_phase = np.angle(np.exp(1j * modified_phase))

    elif method == "Phase noise":
        modified_magnitude = magnitude
        if shared_phase_noise is None:
            rng = np.random.default_rng(int(seed))
            shared_phase_noise = rng.standard_normal(size=phase.shape).astype(np.float32)
        modified_phase = phase + float(phase_noise_std) * shared_phase_noise

    elif method == "Linear phase ramp":
        modified_magnitude = magnitude
        fx = np.fft.fftfreq(w)[None, :]
        fy = np.fft.fftfreq(h)[:, None]
        ramp = -2.0 * np.pi * (float(shift_x) * fx + float(shift_y) * fy)
        modified_phase = phase + ramp

    else:
        raise ValueError("Please select a valid phase modification.")

    modified_spectrum = modified_magnitude * np.exp(1j * modified_phase)
    return modified_spectrum, modified_phase


def apply_phase_modification_rgb(image_rgb, phase_method, quant_levels, phase_noise_std, random_seed, shift_x, shift_y):
    h, w, _ = image_rgb.shape
    shared_random_phase = None
    shared_phase_noise = None

    if phase_method == "Random phase reconstruction":
        rng = np.random.default_rng(int(random_seed))
        shared_random_phase = rng.uniform(-np.pi, np.pi, size=(h, w)).astype(np.float32)

    if phase_method == "Phase noise":
        rng = np.random.default_rng(int(random_seed))
        shared_phase_noise = rng.standard_normal(size=(h, w)).astype(np.float32)

    output = np.zeros_like(image_rgb, dtype=np.float32)
    gray = rgb_to_gray(image_rgb)
    gray_spectrum, modified_phase = _build_modified_spectrum(
        np.fft.fft2(gray),
        phase_method,
        quant_levels,
        phase_noise_std,
        shift_x,
        shift_y,
        random_seed,
        shared_random_phase=shared_random_phase,
        shared_phase_noise=shared_phase_noise,
    )

    for c in range(3):
        channel_spectrum = np.fft.fft2(image_rgb[:, :, c])
        modified_spectrum, _ = _build_modified_spectrum(
            channel_spectrum,
            phase_method,
            quant_levels,
            phase_noise_std,
            shift_x,
            shift_y,
            random_seed,
            shared_random_phase=shared_random_phase,
            shared_phase_noise=shared_phase_noise,
        )
        output[:, :, c] = reconstruct_from_complex_spectrum(modified_spectrum)

    output = normalize_rgb_channels(output)
    modified_phase_display = map_phase_to_display(np.angle(np.fft.fftshift(gray_spectrum)))
    return output, modified_phase_display


def apply_phase_modification(image, phase_method, quant_levels, phase_noise_std, random_seed, shift_x, shift_y):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")
    result_rgb, phase_display = apply_phase_modification_rgb(
        image_rgb,
        phase_method,
        quant_levels,
        phase_noise_std,
        random_seed,
        shift_x,
        shift_y,
    )
    return to_uint8(gray_to_rgb(phase_display)), to_uint8(result_rgb)


def apply_frequency_and_phase_modifications(
    image,
    filter_family,
    filter_name,
    cutoff1,
    cutoff2,
    order,
    notch_u,
    notch_v,
    notch_radius,
    kernel_size,
    sigma_spatial,
    sigma_color,
    phase_method,
    quant_levels,
    phase_noise_std,
    random_seed,
    shift_x,
    shift_y,
):
    filtered_rgb = get_filtered_image_rgb(
        image,
        filter_family,
        filter_name,
        cutoff1,
        cutoff2,
        order,
        notch_u,
        notch_v,
        notch_radius,
        kernel_size,
        sigma_spatial,
        sigma_color,
    )
    result_rgb, _ = apply_phase_modification_rgb(
        filtered_rgb,
        phase_method,
        quant_levels,
        phase_noise_std,
        random_seed,
        shift_x,
        shift_y,
    )
    return to_uint8(result_rgb)


def gaussian_kernel_2d(size, sigma):
    size = odd_cap(size)
    sigma = max(float(sigma), 1e-6)
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def motion_kernel_2d(length, angle_deg):
    length = max(1, int(length))
    size = odd_cap(max(3, length))
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    theta = np.deg2rad(float(angle_deg))
    dx = np.cos(theta)
    dy = np.sin(theta)
    steps = max(2 * size, 8 * length)

    ts = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, steps)
    xs = center + ts * dx
    ys = center + ts * dy
    xi = np.clip(np.round(xs).astype(int), 0, size - 1)
    yi = np.clip(np.round(ys).astype(int), 0, size - 1)
    kernel[yi, xi] = 1.0
    if kernel.sum() <= 0:
        kernel[size // 2, size // 2] = 1.0
    kernel /= kernel.sum()
    return kernel


def build_psf(psf_type, kernel_size, sigma, motion_length, motion_angle):
    if psf_type == "Gaussian PSF":
        return gaussian_kernel_2d(kernel_size, sigma)
    return motion_kernel_2d(motion_length, motion_angle)


def add_gaussian_noise(image_rgb, sigma):
    sigma = max(0.0, float(sigma))
    rng = np.random.default_rng(0)
    noisy = image_rgb + sigma * rng.standard_normal(image_rgb.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def add_salt_pepper_noise(image_rgb, amount):
    amount = max(0.0, float(amount))
    if amount <= 0:
        return image_rgb

    rng = np.random.default_rng(0)
    output = image_rgb.copy()
    h, w, _ = output.shape
    n = int(amount * h * w)

    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    output[ys, xs, :] = 1.0

    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    output[ys, xs, :] = 0.0
    return output


def apply_blur(image_rgb, blur_type, kernel_size, blur_sigma, motion_length, motion_angle):
    if blur_type == "Gaussian blur":
        if int(kernel_size) <= 1 or float(blur_sigma) <= 0:
            return image_rgb.copy(), np.array([[1.0]], dtype=np.float32)
        psf = gaussian_kernel_2d(kernel_size, blur_sigma)
    else:
        psf = motion_kernel_2d(motion_length, motion_angle)

    output = np.zeros_like(image_rgb, dtype=np.float32)
    for channel in range(3):
        output[:, :, channel] = fftconvolve(image_rgb[:, :, channel], psf, mode="same")
    return np.clip(output, 0.0, 1.0), psf


def degrade_image_pipeline(image, gaussian_noise_std, salt_pepper_amount,
                           blur_type, blur_kernel_size, blur_sigma,
                           motion_length, motion_angle):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")

    degraded = add_gaussian_noise(image_rgb, gaussian_noise_std)
    degraded = add_salt_pepper_noise(degraded, salt_pepper_amount)
    degraded, _ = apply_blur(degraded, blur_type, blur_kernel_size, blur_sigma, motion_length, motion_angle)
    return degraded








def apply_degradation_if_needed(image, degradation_enabled,
                                gaussian_noise_std, salt_pepper_amount,
                                blur_type, blur_kernel_size, blur_sigma,
                                motion_length, motion_angle):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise ValueError("Please load an image.")

    if not degradation_enabled:
        return None, None

    degraded = degrade_image_pipeline(
        image_rgb,
        gaussian_noise_std,
        salt_pepper_amount,
        blur_type,
        blur_kernel_size,
        blur_sigma,
        motion_length,
        motion_angle,
    )
    degraded_uint8 = to_uint8(degraded)
    return degraded_uint8, degraded_uint8


def choose_restoration_input(original_image, degradation_enabled, degraded_state,
                             gaussian_noise_std, salt_pepper_amount,
                             blur_type, blur_kernel_size, blur_sigma,
                             motion_length, motion_angle):
    original_rgb = to_float_rgb(original_image)
    if original_rgb is None:
        raise ValueError("Please load an image.")

    if degradation_enabled:
        if degraded_state is not None:
            return to_float_rgb(degraded_state)
        return degrade_image_pipeline(
            original_rgb,
            gaussian_noise_std,
            salt_pepper_amount,
            blur_type,
            blur_kernel_size,
            blur_sigma,
            motion_length,
            motion_angle,
        )
    return original_rgb


def apply_noise_removal(image_rgb, method,
                        median_kernel_size, gaussian_denoise_sigma, tv_weight,
                        bilateral_spatial, bilateral_color,
                        kuwahara_radius,
                        nlm_patch_size, nlm_patch_distance, nlm_h,
                        wavelet_sigma):
    if method == "Median filter":
        kernel_size = odd_cap(median_kernel_size)
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = median_filter(image_rgb[:, :, channel], size=kernel_size)
        return np.clip(output, 0.0, 1.0)

    if method == "Gaussian filter":
        sigma = max(0.1, float(gaussian_denoise_sigma))
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = gaussian_filter(image_rgb[:, :, channel], sigma=sigma)
        return np.clip(output, 0.0, 1.0)

    if method == "Total variation":
        weight = max(0.001, float(tv_weight))
        output = denoise_tv_chambolle(image_rgb, weight=weight, channel_axis=-1)
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    if method == "Bilateral filter":
        output = denoise_bilateral(
            image_rgb,
            sigma_spatial=max(0.1, float(bilateral_spatial)),
            sigma_color=max(0.01, float(bilateral_color)),
            channel_axis=-1,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    if method == "Kuwahara filter":
        return kuwahara_filter_color(image_rgb, kuwahara_radius)

    if method == "Non-local means":
        patch_size = odd_cap(nlm_patch_size)
        patch_distance = max(1, int(nlm_patch_distance))
        h_value = max(0.001, float(nlm_h))
        output = denoise_nl_means(
            image_rgb,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h_value,
            fast_mode=True,
            channel_axis=-1,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    if method == "Wavelet denoise":
        sigma = max(0.001, float(wavelet_sigma))
        output = denoise_wavelet(
            image_rgb,
            sigma=sigma,
            channel_axis=-1,
            method="BayesShrink",
            rescale_sigma=True,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    return image_rgb


def apply_deblurring(image_rgb, method, psf_type,
                     psf_kernel_size, psf_sigma,
                     motion_length, motion_angle,
                     wiener_balance, rl_iterations):
    psf = build_psf(psf_type, psf_kernel_size, psf_sigma, motion_length, motion_angle)
    output = np.zeros_like(image_rgb, dtype=np.float32)

    if method == "Wiener":
        balance = max(0.001, float(wiener_balance))
        for channel in range(3):
            output[:, :, channel] = wiener(image_rgb[:, :, channel], psf, balance=balance)
        return np.clip(output, 0.0, 1.0)

    if method == "Richardson-Lucy":
        num_iter = max(1, int(rl_iterations))
        for channel in range(3):
            output[:, :, channel] = richardson_lucy(image_rgb[:, :, channel], psf, num_iter=num_iter, clip=False)
        return np.clip(output, 0.0, 1.0)

    if method == "Unsupervised Wiener":
        for channel in range(3):
            restored, _ = unsupervised_wiener(image_rgb[:, :, channel], psf, clip=False)
            output[:, :, channel] = restored
        return np.clip(output, 0.0, 1.0)

    return image_rgb


def apply_restoration(original_image, degradation_enabled, degraded_state,
                      gaussian_noise_std, salt_pepper_amount,
                      blur_type, blur_kernel_size, blur_sigma,
                      degr_motion_length, degr_motion_angle,
                      noise_enabled, noise_method,
                      median_kernel_size, gaussian_denoise_sigma, tv_weight,
                      bilateral_spatial, bilateral_color,
                      kuwahara_radius,
                      nlm_patch_size, nlm_patch_distance, nlm_h,
                      wavelet_sigma,
                      deblur_enabled, deblur_method, psf_type,
                      psf_kernel_size, psf_sigma,
                      deblur_motion_length, deblur_motion_angle,
                      wiener_balance, rl_iterations):
    current = choose_restoration_input(
        original_image,
        degradation_enabled,
        degraded_state,
        gaussian_noise_std,
        salt_pepper_amount,
        blur_type,
        blur_kernel_size,
        blur_sigma,
        degr_motion_length,
        degr_motion_angle,
    )

    if noise_enabled:
        current = apply_noise_removal(
            current,
            noise_method,
            median_kernel_size,
            gaussian_denoise_sigma,
            tv_weight,
            bilateral_spatial,
            bilateral_color,
            kuwahara_radius,
            nlm_patch_size,
            nlm_patch_distance,
            nlm_h,
            wavelet_sigma,
        )

    if deblur_enabled:
        current = apply_deblurring(
            current,
            deblur_method,
            psf_type,
            psf_kernel_size,
            psf_sigma,
            deblur_motion_length,
            deblur_motion_angle,
            wiener_balance,
            rl_iterations,
        )

    return to_uint8(current)


def reset_restoration_outputs():
    return None, None, None


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------

PANEL_CSS = """
<style>
.block-container {
    padding-top: 4.75rem !important;
    padding-bottom: 2rem;
}
div[data-testid="stTabs"] {
    margin-top: 0.35rem;
}
.stButton > button {
    width: 100%;
    min-height: 2.45rem;
    white-space: normal;
}
[data-testid="stImage"] img {
    border-radius: 0.35rem;
}
[data-testid="stFileUploader"] {
    margin-bottom: 0.25rem;
}
</style>
"""


def load_streamlit_image(uploaded_file, fallback=DEFAULT_IMAGE):
    if uploaded_file is None:
        return np.array(fallback).copy()
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def clamp_value(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def ensure_widget_value(key, default, minimum, maximum):
    if key not in st.session_state:
        st.session_state[key] = clamp_value(default, minimum, maximum)
    else:
        st.session_state[key] = clamp_value(st.session_state[key], minimum, maximum)


def optional_slider(show, label, minimum, maximum, default, step, key, disabled=False):
    ensure_widget_value(key, default, minimum, maximum)
    if show:
        return st.slider(
            label,
            min_value=minimum,
            max_value=maximum,
            step=step,
            key=key,
            disabled=disabled,
        )
    return st.session_state[key]


def image_signature(image):
    arr = to_float_rgb(image)
    return (arr.shape, float(np.mean(arr)), float(np.std(arr)))


def show_placeholder(label):
    st.info(label)


def display_image_from_state(key, caption, placeholder="Run the corresponding operation to display this result."):
    image = st.session_state.get(key)
    if image is None:
        show_placeholder(placeholder)
    else:
        st.image(image, caption=caption, use_container_width=True)


def safe_run(action, *args, **kwargs):
    try:
        return action(*args, **kwargs), None
    except Exception as exc:
        return None, str(exc)



def filtering_tab():
    uploaded_image = st.file_uploader(
        "Original image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="filtering_uploader",
    )
    image = load_streamlit_image(uploaded_image)
    h, w, max_radius, max_kernel, _, _ = get_image_geometry(image)
    max_notch_u = max(1, w // 2)
    max_notch_v = max(1, h // 2)
    max_notch_radius = max(1, min(h, w) // 2)

    # Same Gradio row: original image + button | magnitude spectrum | phase spectrum
    top_cols = st.columns([1, 1, 1])
    with top_cols[0]:
        st.image(image, caption="Original image", use_container_width=True)
        compute_fourier_clicked = st.button(
            "Compute Fourier transform",
            type="primary",
            key="compute_fourier_btn",
        )
    with top_cols[1]:
        display_image_from_state("centered_spectrum", "Centered magnitude spectrum")
    with top_cols[2]:
        display_image_from_state("phase_plot", "Phase spectrum")

    if compute_fourier_clicked:
        result, error = safe_run(compute_fft_displays, image)
        if error:
            st.error(error)
        else:
            st.session_state["centered_spectrum"] = result[0]
            st.session_state["phase_plot"] = result[1]
            st.rerun()

    # Same Gradio row: controls panel | mask | filtered spectrum | filtered image
    filter_cols = st.columns([1, 1, 1, 1])
    with filter_cols[0]:
        with st.container(border=True):
            filter_family = st.radio(
                "Filter family",
                options=["Linear", "Non-linear"],
                index=0,
                key="filter_family",
            )
            filter_options = LINEAR_FILTERS if filter_family == "Linear" else NONLINEAR_FILTERS
            filter_key = "filter_name_linear" if filter_family == "Linear" else "filter_name_nonlinear"
            filter_name = st.selectbox(
                "Filter",
                options=filter_options,
                index=0,
                key=filter_key,
            )
            apply_filter_clicked = st.button(
                "Apply filter",
                type="primary",
                key="apply_filter_btn",
            )

            show_cutoff1 = filter_family == "Linear" and filter_name in {
                "Low-pass",
                "High-pass",
                "Gaussian low-pass",
                "Gaussian high-pass",
                "Butterworth low-pass",
                "Butterworth high-pass",
            }
            show_cutoff2 = filter_family == "Linear" and filter_name in {"Ideal band-pass", "Ideal band-stop"}
            show_order = filter_family == "Linear" and filter_name in {"Butterworth low-pass", "Butterworth high-pass"}
            show_notch = filter_family == "Linear" and filter_name == "Notch reject"
            show_kernel = filter_family == "Non-linear" and filter_name in {"Median filter", "Minimum filter", "Maximum filter"}
            show_bilateral = filter_family == "Non-linear" and filter_name == "Bilateral filter"

            cutoff1 = optional_slider(
                show_cutoff1 or show_cutoff2,
                "Cutoff radius D₀ (frequency pixels)",
                1,
                max_radius,
                min(30, max_radius),
                1,
                "cutoff1",
            )
            cutoff2 = optional_slider(
                show_cutoff2,
                "Cutoff radius D₁ (frequency pixels)",
                1,
                max_radius,
                min(80, max_radius),
                1,
                "cutoff2",
            )
            order = optional_slider(show_order, "Butterworth order n", 1, 20, 2, 1, "order")
            notch_u = optional_slider(
                show_notch,
                "Notch center u₀ (frequency pixels)",
                -max_notch_u,
                max_notch_u,
                min(32, max_notch_u),
                1,
                "notch_u",
            )
            notch_v = optional_slider(
                show_notch,
                "Notch center v₀ (frequency pixels)",
                -max_notch_v,
                max_notch_v,
                0,
                1,
                "notch_v",
            )
            notch_radius = optional_slider(
                show_notch,
                "Notch radius R (frequency pixels)",
                1,
                max_notch_radius,
                min(10, max_notch_radius),
                1,
                "notch_radius",
            )
            kernel_size = optional_slider(
                show_kernel,
                "Kernel size (odd)",
                1,
                max_kernel,
                min(3, max_kernel),
                1,
                "kernel_size",
            )
            sigma_spatial = optional_slider(
                show_bilateral,
                "Spatial sigma σs (pixels)",
                0.1,
                30.0,
                5.0,
                0.1,
                "sigma_spatial",
            )
            sigma_color = optional_slider(
                show_bilateral,
                "Color sigma σr (intensity)",
                0.01,
                1.0,
                0.1,
                0.01,
                "sigma_color",
            )

    if apply_filter_clicked:
        result, error = safe_run(
            apply_selected_filter,
            image,
            filter_family,
            filter_name,
            cutoff1,
            cutoff2,
            order,
            notch_u,
            notch_v,
            notch_radius,
            kernel_size,
            sigma_spatial,
            sigma_color,
        )
        if error:
            st.error(error)
        else:
            st.session_state["live_mask"] = result[0]
            st.session_state["filtered_spectrum"] = result[1]
            st.session_state["filtered_image"] = result[2]
            st.rerun()

    with filter_cols[1]:
        live_result, error = safe_run(
            update_live_mask,
            image,
            filter_family,
            filter_name,
            cutoff1,
            cutoff2,
            order,
            notch_u,
            notch_v,
            notch_radius,
        )
        if error:
            st.error(error)
        else:
            st.image(live_result, caption="Frequency mask", use_container_width=True)
    with filter_cols[2]:
        display_image_from_state("filtered_spectrum", "Filtered image spectrum")
    with filter_cols[3]:
        display_image_from_state("filtered_image", "Filtered image")

    # Same Gradio row: phase controls panel | modified phase | phase-modified image
    phase_cols = st.columns([1, 1, 1])
    with phase_cols[0]:
        with st.container(border=True):
            phase_method = st.selectbox(
                "Phase modification",
                options=PHASE_MODIFICATION_METHODS,
                index=0,
                key="phase_method",
            )
            quant_levels = optional_slider(
                phase_method == "Phase quantization",
                "Quantization levels L",
                2,
                64,
                8,
                1,
                "quant_levels",
            )
            phase_noise_std = optional_slider(
                phase_method == "Phase noise",
                "Phase noise std σφ (rad)",
                0.0,
                float(np.pi),
                0.5,
                0.01,
                "phase_noise_std",
            )
            random_seed = optional_slider(
                phase_method in {"Random phase reconstruction", "Phase noise"},
                "Random seed",
                0,
                9999,
                0,
                1,
                "random_seed",
            )
            shift_x = optional_slider(
                phase_method == "Linear phase ramp",
                "Shift Δx (pixels)",
                -w,
                w,
                0,
                1,
                "shift_x",
            )
            shift_y = optional_slider(
                phase_method == "Linear phase ramp",
                "Shift Δy (pixels)",
                -h,
                h,
                0,
                1,
                "shift_y",
            )
            apply_phase_clicked = st.button(
                "Apply phase modification",
                type="primary",
                key="apply_phase_btn",
            )

    if apply_phase_clicked:
        result, error = safe_run(
            apply_phase_modification,
            image,
            phase_method,
            quant_levels,
            phase_noise_std,
            random_seed,
            shift_x,
            shift_y,
        )
        if error:
            st.error(error)
        else:
            st.session_state["modified_phase_plot"] = result[0]
            st.session_state["phase_result_image"] = result[1]
            st.rerun()

    with phase_cols[1]:
        display_image_from_state("modified_phase_plot", "Phase spectrum after modification")
    with phase_cols[2]:
        display_image_from_state("phase_result_image", "Phase-modified image")

    # Same Gradio row: button panel | combined result
    combined_cols = st.columns([1, 2])
    with combined_cols[0]:
        with st.container(border=True):
            apply_both_clicked = st.button(
                "Apply filter + phase modification",
                type="primary",
                key="apply_both_btn",
            )
    if apply_both_clicked:
        result, error = safe_run(
            apply_frequency_and_phase_modifications,
            image,
            filter_family,
            filter_name,
            cutoff1,
            cutoff2,
            order,
            notch_u,
            notch_v,
            notch_radius,
            kernel_size,
            sigma_spatial,
            sigma_color,
            phase_method,
            quant_levels,
            phase_noise_std,
            random_seed,
            shift_x,
            shift_y,
        )
        if error:
            st.error(error)
        else:
            st.session_state["combined_result_image"] = result
            st.rerun()
    with combined_cols[1]:
        display_image_from_state("combined_result_image", "Combined result image")


def restoration_tab():
    uploaded_image = st.file_uploader(
        "Original image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="restoration_uploader",
    )
    image = load_streamlit_image(uploaded_image)
    _, _, _, max_kernel, max_motion, max_kuwahara = get_image_geometry(image)

    current_signature = image_signature(image)
    if st.session_state.get("restoration_image_signature") != current_signature:
        st.session_state["restoration_image_signature"] = current_signature
        st.session_state["degraded_image"] = None
        st.session_state["degraded_state"] = None
        st.session_state["restored_image"] = None

    # Same Gradio row: original image | degradation panel containing controls + degraded image
    top_cols = st.columns([1, 1])
    with top_cols[0]:
        st.image(image, caption="Original image", use_container_width=True)

    with top_cols[1]:
        with st.container(border=True):
            use_degradation = st.checkbox("Enable degradation", value=False, key="use_degradation")
            degradation_cols = st.columns([1, 1])
            with degradation_cols[0]:
                gaussian_noise_std = st.slider(
                    "Gaussian noise std σn",
                    0.0,
                    1.0,
                    0.03,
                    0.001,
                    key="gaussian_noise_std",
                    disabled=not use_degradation,
                )
                salt_pepper_amount = st.slider(
                    "Salt-and-pepper density ρ",
                    0.0,
                    0.5,
                    0.02,
                    0.001,
                    key="salt_pepper_amount",
                    disabled=not use_degradation,
                )
                blur_type = st.selectbox(
                    "Blur type",
                    options=BLUR_TYPES,
                    index=0,
                    key="blur_type",
                    disabled=not use_degradation,
                )
                blur_kernel_size = optional_slider(
                    blur_type == "Gaussian blur",
                    "Gaussian PSF kernel size",
                    1,
                    max_kernel,
                    min(9, max_kernel),
                    1,
                    "blur_kernel_size",
                    disabled=not use_degradation,
                )
                blur_sigma = optional_slider(
                    blur_type == "Gaussian blur",
                    "Gaussian PSF sigma σ",
                    0.1,
                    20.0,
                    2.0,
                    0.1,
                    "blur_sigma",
                    disabled=not use_degradation,
                )
                degr_motion_length = optional_slider(
                    blur_type == "Motion blur",
                    "Motion PSF length L (pixels)",
                    1,
                    max_motion,
                    min(15, max_motion),
                    1,
                    "degr_motion_length",
                    disabled=not use_degradation,
                )
                degr_motion_angle = optional_slider(
                    blur_type == "Motion blur",
                    "Motion PSF angle θ (degrees)",
                    -180.0,
                    180.0,
                    0.0,
                    1.0,
                    "degr_motion_angle",
                    disabled=not use_degradation,
                )
                apply_degradation_clicked = st.button(
                    "Apply degradation",
                    type="primary",
                    key="apply_degradation_btn",
                    disabled=not use_degradation,
                )
            with degradation_cols[1]:
                display_image_from_state(
                    "degraded_image",
                    "Degraded image",
                    "Enable degradation and run it to display the degraded image.",
                )

    if apply_degradation_clicked:
        result, error = safe_run(
            apply_degradation_if_needed,
            image,
            use_degradation,
            gaussian_noise_std,
            salt_pepper_amount,
            blur_type,
            blur_kernel_size,
            blur_sigma,
            degr_motion_length,
            degr_motion_angle,
        )
        if error:
            st.error(error)
        else:
            st.session_state["degraded_image"] = result[0]
            st.session_state["degraded_state"] = result[1]
            st.rerun()

    # Same Gradio row: noise removal panel | deblurring panel
    method_cols = st.columns([1, 1])
    with method_cols[0]:
        with st.container(border=True):
            use_noise_removal = st.checkbox("Enable noise removal", value=False, key="use_noise_removal")
            noise_method = st.selectbox(
                "Noise removal method",
                options=NOISE_REMOVAL_FILTERS,
                index=0,
                key="noise_method",
                disabled=not use_noise_removal,
            )
            median_kernel_size = optional_slider(
                noise_method == "Median filter",
                "Median kernel size (odd)",
                1,
                max_kernel,
                min(3, max_kernel),
                1,
                "median_kernel_size",
                disabled=not use_noise_removal,
            )
            gaussian_denoise_sigma = optional_slider(
                noise_method == "Gaussian filter",
                "Gaussian sigma σ",
                0.1,
                20.0,
                1.0,
                0.1,
                "gaussian_denoise_sigma",
                disabled=not use_noise_removal,
            )
            tv_weight = optional_slider(
                noise_method == "Total variation",
                "TV weight λ",
                0.001,
                1.0,
                0.08,
                0.001,
                "tv_weight",
                disabled=not use_noise_removal,
            )
            bilateral_spatial = optional_slider(
                noise_method == "Bilateral filter",
                "Bilateral spatial sigma σs",
                0.1,
                30.0,
                5.0,
                0.1,
                "bilateral_spatial",
                disabled=not use_noise_removal,
            )
            bilateral_color = optional_slider(
                noise_method == "Bilateral filter",
                "Bilateral color sigma σr",
                0.01,
                1.0,
                0.1,
                0.01,
                "bilateral_color",
                disabled=not use_noise_removal,
            )
            kuwahara_radius = optional_slider(
                noise_method == "Kuwahara filter",
                "Kuwahara radius r",
                1,
                max_kuwahara,
                min(2, max_kuwahara),
                1,
                "kuwahara_radius",
                disabled=not use_noise_removal,
            )
            nlm_patch_size = optional_slider(
                noise_method == "Non-local means",
                "NLM patch size p",
                1,
                11,
                5,
                1,
                "nlm_patch_size",
                disabled=not use_noise_removal,
            )
            nlm_patch_distance = optional_slider(
                noise_method == "Non-local means",
                "NLM patch search distance",
                1,
                20,
                6,
                1,
                "nlm_patch_distance",
                disabled=not use_noise_removal,
            )
            nlm_h = optional_slider(
                noise_method == "Non-local means",
                "NLM filter parameter h",
                0.001,
                1.0,
                0.08,
                0.001,
                "nlm_h",
                disabled=not use_noise_removal,
            )
            wavelet_sigma = optional_slider(
                noise_method == "Wavelet denoise",
                "Wavelet noise sigma σn",
                0.001,
                1.0,
                0.08,
                0.001,
                "wavelet_sigma",
                disabled=not use_noise_removal,
            )
            apply_noise_clicked = st.button(
                "Apply noise removal",
                type="primary",
                key="apply_noise_btn",
                disabled=not use_noise_removal,
            )

    with method_cols[1]:
        with st.container(border=True):
            use_deblurring = st.checkbox("Enable deblurring", value=False, key="use_deblurring")
            deblur_method = st.selectbox(
                "Deblurring method",
                options=DEBLUR_METHODS,
                index=0,
                key="deblur_method",
                disabled=not use_deblurring,
            )
            psf_type = st.selectbox(
                "PSF type",
                options=["Gaussian PSF", "Motion PSF"],
                index=0,
                key="psf_type",
                disabled=not use_deblurring,
            )
            psf_kernel_size = optional_slider(
                psf_type == "Gaussian PSF",
                "Gaussian PSF kernel size",
                1,
                max_kernel,
                min(9, max_kernel),
                1,
                "psf_kernel_size",
                disabled=not use_deblurring,
            )
            psf_sigma = optional_slider(
                psf_type == "Gaussian PSF",
                "Gaussian PSF sigma σ",
                0.1,
                20.0,
                2.0,
                0.1,
                "psf_sigma",
                disabled=not use_deblurring,
            )
            deblur_motion_length = optional_slider(
                psf_type == "Motion PSF",
                "Motion PSF length L (pixels)",
                1,
                max_motion,
                min(15, max_motion),
                1,
                "deblur_motion_length",
                disabled=not use_deblurring,
            )
            deblur_motion_angle = optional_slider(
                psf_type == "Motion PSF",
                "Motion PSF angle θ (degrees)",
                -180.0,
                180.0,
                0.0,
                1.0,
                "deblur_motion_angle",
                disabled=not use_deblurring,
            )
            wiener_balance = optional_slider(
                deblur_method in {"Wiener", "Unsupervised Wiener"},
                "Wiener balance K (NSR)",
                0.001,
                10.0,
                0.05,
                0.001,
                "wiener_balance",
                disabled=not use_deblurring,
            )
            rl_iterations = optional_slider(
                deblur_method == "Richardson-Lucy",
                "Richardson-Lucy iterations",
                1,
                100,
                20,
                1,
                "rl_iterations",
                disabled=not use_deblurring,
            )
            apply_deblur_clicked = st.button(
                "Apply deblurring",
                type="primary",
                key="apply_deblur_btn",
                disabled=not use_deblurring,
            )

    restoration_inputs = dict(
        original_image=image,
        degradation_enabled=use_degradation,
        degraded_state=st.session_state.get("degraded_state"),
        gaussian_noise_std=gaussian_noise_std,
        salt_pepper_amount=salt_pepper_amount,
        blur_type=blur_type,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        degr_motion_length=degr_motion_length,
        degr_motion_angle=degr_motion_angle,
        noise_enabled=use_noise_removal,
        noise_method=noise_method,
        median_kernel_size=median_kernel_size,
        gaussian_denoise_sigma=gaussian_denoise_sigma,
        tv_weight=tv_weight,
        bilateral_spatial=bilateral_spatial,
        bilateral_color=bilateral_color,
        kuwahara_radius=kuwahara_radius,
        nlm_patch_size=nlm_patch_size,
        nlm_patch_distance=nlm_patch_distance,
        nlm_h=nlm_h,
        wavelet_sigma=wavelet_sigma,
        deblur_enabled=use_deblurring,
        deblur_method=deblur_method,
        psf_type=psf_type,
        psf_kernel_size=psf_kernel_size,
        psf_sigma=psf_sigma,
        deblur_motion_length=deblur_motion_length,
        deblur_motion_angle=deblur_motion_angle,
        wiener_balance=wiener_balance,
        rl_iterations=rl_iterations,
    )

    if apply_noise_clicked:
        result, error = safe_run(apply_restoration, **restoration_inputs)
        if error:
            st.error(error)
        else:
            st.session_state["restored_image"] = result
            st.rerun()

    if apply_deblur_clicked:
        result, error = safe_run(apply_restoration, **restoration_inputs)
        if error:
            st.error(error)
        else:
            st.session_state["restored_image"] = result
            st.rerun()

    display_image_from_state("restored_image", "Restored image")


def set_doc_section(state_key, title):
    st.session_state[state_key] = title


def documentation_tab(sections, state_key, missing_message):
    if not sections:
        st.warning(missing_message)
        return

    titles = list(sections.keys())
    if state_key not in st.session_state or st.session_state[state_key] not in sections:
        st.session_state[state_key] = titles[0]

    # Same Gradio row: navigation buttons | markdown viewer.
    nav_col, view_col = st.columns([1, 2])
    with nav_col:
        for idx, title in enumerate(titles):
            selected = st.session_state[state_key] == title
            st.button(
                title,
                key=f"{state_key}_doc_button_{idx}",
                type="primary" if selected else "secondary",
                use_container_width=True,
                on_click=set_doc_section,
                args=(state_key, title),
            )
    with view_col:
        st.markdown(sections[st.session_state[state_key]])


def main():
    st.set_page_config(page_title="Fourier Image Lab", layout="wide")
    st.markdown(PANEL_CSS, unsafe_allow_html=True)

    tabs = st.tabs(["Filtering", "Denoising & Deblurring", "Documentation FR", "Documentation EN"])
    with tabs[0]:
        filtering_tab()
    with tabs[1]:
        restoration_tab()
    with tabs[2]:
        documentation_tab(
            DOC_FR_SECTIONS,
            "selected_doc_fr",
            "documentation_fr.md was not found next to app_sl.py.",
        )
    with tabs[3]:
        documentation_tab(
            DOC_EN_SECTIONS,
            "selected_doc_en",
            "documentation_en.md was not found next to app_sl.py.",
        )


if __name__ == "__main__":
    main()
