import re
import gradio as gr
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

with open("documentation_fr.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_fr = f.read()

with open("documentation_en.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_en = f.read()


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


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)

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


def update_filter_ranges(image):
    h, w, max_radius, max_kernel, _, _ = get_image_geometry(image)
    max_notch_u = max(1, w // 2)
    max_notch_v = max(1, h // 2)
    max_notch_radius = max(1, min(h, w) // 2)

    cutoff1_default = min(30, max_radius)
    cutoff2_default = min(80, max_radius)
    notch_u_default = min(32, max_notch_u)
    notch_radius_default = min(10, max_notch_radius)

    return (
        gr.update(minimum=1, maximum=max_radius, value=cutoff1_default, step=1),
        gr.update(minimum=1, maximum=max_radius, value=cutoff2_default, step=1),
        gr.update(minimum=-max_notch_u, maximum=max_notch_u, value=notch_u_default, step=1),
        gr.update(minimum=-max_notch_v, maximum=max_notch_v, value=0, step=1),
        gr.update(minimum=1, maximum=max_notch_radius, value=notch_radius_default, step=1),
        gr.update(minimum=1, maximum=max_kernel, value=min(3, max_kernel), step=1),
    )


def update_phase_ranges(image):
    h, w, _, _, _, _ = get_image_geometry(image)
    return (
        gr.update(minimum=-w, maximum=w, value=0, step=1),
        gr.update(minimum=-h, maximum=h, value=0, step=1),
    )


def update_restoration_ranges(image):
    _, _, _, max_kernel, max_motion, max_kuwahara = get_image_geometry(image)
    blur_default = min(9, max_kernel)
    median_default = min(3, max_kernel)
    psf_default = min(9, max_kernel)
    motion_default = min(15, max_motion)
    kuwahara_default = min(2, max_kuwahara)

    return (
        gr.update(minimum=1, maximum=max_kernel, value=blur_default, step=1),
        gr.update(minimum=1, maximum=max_kernel, value=median_default, step=1),
        gr.update(minimum=1, maximum=max_kernel, value=psf_default, step=1),
        gr.update(minimum=1, maximum=max_motion, value=motion_default, step=1),
        gr.update(minimum=1, maximum=max_motion, value=motion_default, step=1),
        gr.update(minimum=1, maximum=max_kuwahara, value=kuwahara_default, step=1),
    )


def frequency_grid(h, w):
    y = np.arange(h) - h // 2
    x = np.arange(w) - w // 2
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    return xx, yy, d


def compute_fft_displays(image):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise gr.Error("Please load an image.")

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
            sigma_spatial=max(0.01, float(sigma_spatial)),
            sigma_color=max(0.0001, float(sigma_color)),
            channel_axis=-1,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    return image_rgb


def get_filtered_image_rgb(image, filter_family, filter_name,
                           cutoff1, cutoff2, order, notch_u, notch_v, notch_radius,
                           kernel_size, sigma_spatial, sigma_color):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise gr.Error("Please load an image.")

    if filter_family == "Linear" and filter_name in LINEAR_FILTERS:
        h, w, _ = image_rgb.shape
        mask = build_linear_mask(filter_name, h, w, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius)
        return apply_frequency_mask_rgb(image_rgb, mask)

    if filter_family == "Non-linear" and filter_name in NONLINEAR_FILTERS:
        return apply_nonlinear_filter(image_rgb, filter_name, kernel_size, sigma_spatial, sigma_color)

    raise gr.Error("Please select a valid filter family and filter.")


def update_filter_controls(filter_family):
    if filter_family == "Linear":
        return gr.update(choices=LINEAR_FILTERS, value=LINEAR_FILTERS[0], interactive=True)
    if filter_family == "Non-linear":
        return gr.update(choices=NONLINEAR_FILTERS, value=NONLINEAR_FILTERS[0], interactive=True)
    return gr.update(choices=[], value=None, interactive=False)


def update_parameter_visibility(filter_family, filter_name):
    show_cutoff1 = False
    show_cutoff2 = False
    show_order = False
    show_notch_u = False
    show_notch_v = False
    show_notch_radius = False
    show_kernel = False
    show_sigma_spatial = False
    show_sigma_color = False

    if filter_family == "Linear" and filter_name is not None:
        if filter_name in {
            "Low-pass",
            "High-pass",
            "Gaussian low-pass",
            "Gaussian high-pass",
            "Butterworth low-pass",
            "Butterworth high-pass",
        }:
            show_cutoff1 = True
        if filter_name in {"Ideal band-pass", "Ideal band-stop"}:
            show_cutoff1 = True
            show_cutoff2 = True
        if filter_name in {"Butterworth low-pass", "Butterworth high-pass"}:
            show_order = True
        if filter_name == "Notch reject":
            show_notch_u = True
            show_notch_v = True
            show_notch_radius = True

    if filter_family == "Non-linear" and filter_name is not None:
        if filter_name in {"Median filter", "Minimum filter", "Maximum filter"}:
            show_kernel = True
        if filter_name == "Bilateral filter":
            show_sigma_spatial = True
            show_sigma_color = True

    return (
        gr.update(visible=show_cutoff1),
        gr.update(visible=show_cutoff2),
        gr.update(visible=show_order),
        gr.update(visible=show_notch_u),
        gr.update(visible=show_notch_v),
        gr.update(visible=show_notch_radius),
        gr.update(visible=show_kernel),
        gr.update(visible=show_sigma_spatial),
        gr.update(visible=show_sigma_color),
    )


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
        raise gr.Error("Please load an image.")

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
        raise gr.Error("Please select a valid filter family and filter.")

    result_gray = rgb_to_gray(result)
    result_fft = np.fft.fftshift(np.fft.fft2(result_gray))
    result_spectrum = normalize01(np.log1p(np.abs(result_fft)))
    return to_uint8(mask_rgb), to_uint8(gray_to_rgb(result_spectrum)), to_uint8(result)


def update_phase_parameter_visibility(phase_method):
    return (
        gr.update(visible=phase_method == "Phase quantization"),
        gr.update(visible=phase_method == "Phase noise"),
        gr.update(visible=phase_method in {"Random phase reconstruction", "Phase noise"}),
        gr.update(visible=phase_method == "Linear phase ramp"),
        gr.update(visible=phase_method == "Linear phase ramp"),
    )


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
        raise gr.Error("Please select a valid phase modification.")

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
        raise gr.Error("Please load an image.")
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
        raise gr.Error("Please load an image.")

    degraded = add_gaussian_noise(image_rgb, gaussian_noise_std)
    degraded = add_salt_pepper_noise(degraded, salt_pepper_amount)
    degraded, _ = apply_blur(degraded, blur_type, blur_kernel_size, blur_sigma, motion_length, motion_angle)
    return degraded


def update_degradation_controls(enabled):
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def update_degradation_parameter_visibility(blur_type):
    return (
        gr.update(visible=blur_type == "Gaussian blur"),
        gr.update(visible=blur_type == "Gaussian blur"),
        gr.update(visible=blur_type == "Motion blur"),
        gr.update(visible=blur_type == "Motion blur"),
    )


def update_noise_controls(enabled):
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def update_noise_parameter_visibility(method):
    return (
        gr.update(visible=method == "Median filter"),
        gr.update(visible=method == "Gaussian filter"),
        gr.update(visible=method == "Total variation"),
        gr.update(visible=method == "Bilateral filter"),
        gr.update(visible=method == "Bilateral filter"),
        gr.update(visible=method == "Kuwahara filter"),
        gr.update(visible=method == "Non-local means"),
        gr.update(visible=method == "Non-local means"),
        gr.update(visible=method == "Non-local means"),
        gr.update(visible=method == "Wavelet denoise"),
    )


def update_deblur_controls(enabled):
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def update_deblur_parameter_visibility(method, psf_type):
    show_wiener_balance = method in {"Wiener", "Unsupervised Wiener"}
    show_rl_iterations = method == "Richardson-Lucy"
    show_gaussian = psf_type == "Gaussian PSF"
    show_motion = psf_type == "Motion PSF"
    return (
        gr.update(visible=show_gaussian),
        gr.update(visible=show_gaussian),
        gr.update(visible=show_motion),
        gr.update(visible=show_motion),
        gr.update(visible=show_wiener_balance),
        gr.update(visible=show_rl_iterations),
    )


def apply_degradation_if_needed(image, degradation_enabled,
                                gaussian_noise_std, salt_pepper_amount,
                                blur_type, blur_kernel_size, blur_sigma,
                                motion_length, motion_angle):
    image_rgb = to_float_rgb(image)
    if image_rgb is None:
        raise gr.Error("Please load an image.")

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
        raise gr.Error("Please load an image.")

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
        sigma = max(0.01, float(gaussian_denoise_sigma))
        output = np.zeros_like(image_rgb, dtype=np.float32)
        for channel in range(3):
            output[:, :, channel] = gaussian_filter(image_rgb[:, :, channel], sigma=sigma)
        return np.clip(output, 0.0, 1.0)

    if method == "Total variation":
        weight = max(0.0001, float(tv_weight))
        output = denoise_tv_chambolle(image_rgb, weight=weight, channel_axis=-1)
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    if method == "Bilateral filter":
        output = denoise_bilateral(
            image_rgb,
            sigma_spatial=max(0.01, float(bilateral_spatial)),
            sigma_color=max(0.0001, float(bilateral_color)),
            channel_axis=-1,
        )
        return np.clip(output.astype(np.float32), 0.0, 1.0)

    if method == "Kuwahara filter":
        return kuwahara_filter_color(image_rgb, kuwahara_radius)

    if method == "Non-local means":
        patch_size = odd_cap(nlm_patch_size)
        patch_distance = max(1, int(nlm_patch_distance))
        h_value = max(0.0001, float(nlm_h))
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
        sigma = max(0.0001, float(wavelet_sigma))
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
        balance = max(0.0001, float(wiener_balance))
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


with gr.Blocks() as demo:
    with gr.Tab("Filtering"):
        with gr.Row():
            with gr.Column():
                fft_input = gr.Image(value=DEFAULT_IMAGE, type="numpy", label="Original image", height=320)
                compute_fourier_btn = gr.Button("Fourier transform", variant="primary")
            centered_spectrum = gr.Image(label="Centered spectrum", height=360)
            phase_plot = gr.Image(label="Phase", height=360)

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                filter_family = gr.Radio(
                    choices=["Linear", "Non-linear"],
                    value="Linear",
                    label="Filter family",
                )
                filter_name = gr.Dropdown(
                    choices=LINEAR_FILTERS,
                    value=LINEAR_FILTERS[0],
                    label="Filter",
                    interactive=True,
                )
                apply_filter_btn = gr.Button("Apply filter", variant="primary")

                cutoff1 = gr.Slider(1, 100, value=30, step=1, label="Cutoff 1")
                cutoff2 = gr.Slider(1, 100, value=80, step=1, label="Cutoff 2", visible=False)
                order = gr.Slider(1, 20, value=2, step=1, label="Butterworth order", visible=False)
                notch_u = gr.Slider(-100, 100, value=32, step=1, label="Notch offset u", visible=False)
                notch_v = gr.Slider(-100, 100, value=0, step=1, label="Notch offset v", visible=False)
                notch_radius = gr.Slider(1, 100, value=10, step=1, label="Notch radius", visible=False)
                kernel_size = gr.Slider(1, 21, value=3, step=1, label="Kernel size", visible=False)
                sigma_spatial = gr.Slider(0.01, 100.0, value=5.0, step=0.01, label="Spatial sigma", visible=False)
                sigma_color = gr.Slider(0.0001, 1.0, value=0.1, step=0.0001, label="Color sigma", visible=False)

            live_mask = gr.Image(label="Mask", height=360)
            filtered_spectrum = gr.Image(label="Filtered Fourier domain", height=360)
            filtered_image = gr.Image(label="Result image", height=360)

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                phase_method = gr.Dropdown(
                    choices=PHASE_MODIFICATION_METHODS,
                    value=PHASE_MODIFICATION_METHODS[0],
                    label="Phase modification",
                    interactive=True,
                )
                quant_levels = gr.Slider(2, 64, value=8, step=1, label="Quantization levels", visible=False)
                phase_noise_std = gr.Slider(0.0, np.pi, value=0.5, step=0.01, label="Phase noise std (rad)", visible=False)
                random_seed = gr.Slider(0, 9999, value=0, step=1, label="Random seed", visible=False)
                shift_x = gr.Slider(-512, 512, value=0, step=1, label="Shift x (pixels)", visible=False)
                shift_y = gr.Slider(-512, 512, value=0, step=1, label="Shift y (pixels)", visible=False)
                apply_phase_btn = gr.Button("Apply phase modification", variant="primary")

            modified_phase_plot = gr.Image(label="Phase after modification", height=360)
            phase_result_image = gr.Image(label="Result image", height=360)

        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                apply_both_btn = gr.Button("Apply frequency and phase modifications", variant="primary")
            combined_result_image = gr.Image(label="Combined result image", height=360)

        compute_fourier_btn.click(
            fn=compute_fft_displays,
            inputs=[fft_input],
            outputs=[centered_spectrum, phase_plot],
        )

        filter_family.change(
            fn=update_filter_controls,
            inputs=[filter_family],
            outputs=[filter_name],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        filter_family.change(
            fn=update_parameter_visibility,
            inputs=[filter_family, filter_name],
            outputs=[cutoff1, cutoff2, order, notch_u, notch_v, notch_radius, kernel_size, sigma_spatial, sigma_color],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )
        filter_name.change(
            fn=update_parameter_visibility,
            inputs=[filter_family, filter_name],
            outputs=[cutoff1, cutoff2, order, notch_u, notch_v, notch_radius, kernel_size, sigma_spatial, sigma_color],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        live_mask_inputs = [
            fft_input, filter_family, filter_name,
            cutoff1, cutoff2, order, notch_u, notch_v, notch_radius,
        ]
        for component in [fft_input, filter_family, filter_name, cutoff1, cutoff2, order, notch_u, notch_v, notch_radius]:
            component.change(
                fn=update_live_mask,
                inputs=live_mask_inputs,
                outputs=[live_mask],
                queue=False,
                show_progress="hidden",
                trigger_mode="once",
            )

        apply_filter_btn.click(
            fn=apply_selected_filter,
            inputs=[
                fft_input, filter_family, filter_name,
                cutoff1, cutoff2, order, notch_u, notch_v, notch_radius,
                kernel_size, sigma_spatial, sigma_color,
            ],
            outputs=[live_mask, filtered_spectrum, filtered_image],
        )

        phase_method.change(
            fn=update_phase_parameter_visibility,
            inputs=[phase_method],
            outputs=[quant_levels, phase_noise_std, random_seed, shift_x, shift_y],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        apply_phase_btn.click(
            fn=apply_phase_modification,
            inputs=[fft_input, phase_method, quant_levels, phase_noise_std, random_seed, shift_x, shift_y],
            outputs=[modified_phase_plot, phase_result_image],
        )

        apply_both_btn.click(
            fn=apply_frequency_and_phase_modifications,
            inputs=[
                fft_input,
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
            ],
            outputs=[combined_result_image],
        )

        fft_input.change(
            fn=update_filter_ranges,
            inputs=[fft_input],
            outputs=[cutoff1, cutoff2, notch_u, notch_v, notch_radius, kernel_size],
            queue=False,
        )
        fft_input.change(
            fn=update_phase_ranges,
            inputs=[fft_input],
            outputs=[shift_x, shift_y],
            queue=False,
        )

    with gr.Tab("Denoising and deblurring"):
        degraded_state = gr.State(value=None)

        with gr.Row():
            restore_input = gr.Image(value=DEFAULT_IMAGE, type="numpy", label="Original image", height=340)

            with gr.Column(variant="panel"):
                use_degradation = gr.Checkbox(label="Enable degradation", value=False)
                with gr.Row():
                    with gr.Column():
                        gaussian_noise_std = gr.Slider(
                            0.0, 1.0, value=0.03, step=0.001,
                            label="Gaussian noise std", interactive=False
                        )
                        salt_pepper_amount = gr.Slider(
                            0.0, 0.5, value=0.02, step=0.001,
                            label="Salt and pepper amount", interactive=False
                        )
                        blur_type = gr.Dropdown(
                            choices=BLUR_TYPES,
                            value=BLUR_TYPES[0],
                            label="Blur type",
                            interactive=False,
                        )
                        blur_kernel_size = gr.Slider(
                            1, 21, value=9, step=1,
                            label="Gaussian kernel size", interactive=False
                        )
                        blur_sigma = gr.Slider(
                            0.0, 100.0, value=2.0, step=0.01,
                            label="Gaussian sigma", interactive=False
                        )
                        degr_motion_length = gr.Slider(
                            1, 101, value=15, step=1,
                            label="Motion length", visible=False, interactive=False
                        )
                        degr_motion_angle = gr.Slider(
                            -180.0, 180.0, value=0.0, step=1.0,
                            label="Motion angle (degrees)", visible=False, interactive=False
                        )
                        apply_degradation_btn = gr.Button("Apply degradation", variant="primary", interactive=False)
                    degraded_image = gr.Image(label="Degraded image", height=300)

        with gr.Row():
            with gr.Column(variant="panel"):
                use_noise_removal = gr.Checkbox(label="Enable noise removal", value=False)
                noise_method = gr.Dropdown(
                    choices=NOISE_REMOVAL_FILTERS,
                    value=NOISE_REMOVAL_FILTERS[0],
                    label="Noise removal method",
                    interactive=False,
                )
                median_kernel_size = gr.Slider(
                    1, 21, value=3, step=1,
                    label="Median kernel size", visible=True, interactive=False
                )
                gaussian_denoise_sigma = gr.Slider(
                    0.01, 100.0, value=1.0, step=0.01,
                    label="Gaussian sigma", visible=False, interactive=False
                )
                tv_weight = gr.Slider(
                    0.0001, 10.0, value=0.08, step=0.0001,
                    label="TV weight", visible=False, interactive=False
                )
                bilateral_spatial = gr.Slider(
                    0.01, 100.0, value=5.0, step=0.01,
                    label="Bilateral spatial sigma", visible=False, interactive=False
                )
                bilateral_color = gr.Slider(
                    0.0001, 1.0, value=0.1, step=0.0001,
                    label="Bilateral color sigma", visible=False, interactive=False
                )
                kuwahara_radius = gr.Slider(
                    1, 32, value=2, step=1,
                    label="Kuwahara radius", visible=False, interactive=False
                )
                nlm_patch_size = gr.Slider(
                    1, 31, value=5, step=1,
                    label="NLM patch size", visible=False, interactive=False
                )
                nlm_patch_distance = gr.Slider(
                    1, 50, value=6, step=1,
                    label="NLM patch distance", visible=False, interactive=False
                )
                nlm_h = gr.Slider(
                    0.0001, 1.0, value=0.08, step=0.0001,
                    label="NLM h", visible=False, interactive=False
                )
                wavelet_sigma = gr.Slider(
                    0.0001, 1.0, value=0.08, step=0.0001,
                    label="Wavelet sigma", visible=False, interactive=False
                )
                apply_noise_btn = gr.Button("Apply noise removal", variant="primary", interactive=False)

            with gr.Column(variant="panel"):
                use_deblurring = gr.Checkbox(label="Enable deblurring", value=False)
                deblur_method = gr.Dropdown(
                    choices=DEBLUR_METHODS,
                    value=DEBLUR_METHODS[0],
                    label="Deblurring method",
                    interactive=False,
                )
                psf_type = gr.Dropdown(
                    choices=["Gaussian PSF", "Motion PSF"],
                    value="Gaussian PSF",
                    label="PSF type",
                    interactive=False,
                )
                psf_kernel_size = gr.Slider(
                    1, 21, value=9, step=1,
                    label="Gaussian PSF kernel size", interactive=False
                )
                psf_sigma = gr.Slider(
                    0.01, 100.0, value=2.0, step=0.01,
                    label="Gaussian PSF sigma", interactive=False
                )
                deblur_motion_length = gr.Slider(
                    1, 101, value=15, step=1,
                    label="Motion PSF length", visible=False, interactive=False
                )
                deblur_motion_angle = gr.Slider(
                    -180.0, 180.0, value=0.0, step=1.0,
                    label="Motion PSF angle (degrees)", visible=False, interactive=False
                )
                wiener_balance = gr.Slider(
                    0.0001, 10.0, value=0.05, step=0.0001,
                    label="Wiener balance", interactive=False
                )
                rl_iterations = gr.Slider(
                    1, 200, value=20, step=1,
                    label="Richardson-Lucy iterations", visible=False, interactive=False
                )
                apply_deblur_btn = gr.Button("Apply deblurring", variant="primary", interactive=False)

            restored_image = gr.Image(label="Result image", height=340)

        use_degradation.change(
            fn=update_degradation_controls,
            inputs=[use_degradation],
            outputs=[
                gaussian_noise_std,
                salt_pepper_amount,
                blur_type,
                blur_kernel_size,
                blur_sigma,
                degr_motion_length,
                degr_motion_angle,
                apply_degradation_btn,
            ],
            queue=False,
        )

        blur_type.change(
            fn=update_degradation_parameter_visibility,
            inputs=[blur_type],
            outputs=[blur_kernel_size, blur_sigma, degr_motion_length, degr_motion_angle],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        use_noise_removal.change(
            fn=update_noise_controls,
            inputs=[use_noise_removal],
            outputs=[
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
                apply_noise_btn,
            ],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        noise_method.change(
            fn=update_noise_parameter_visibility,
            inputs=[noise_method],
            outputs=[
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
            ],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        use_deblurring.change(
            fn=update_deblur_controls,
            inputs=[use_deblurring],
            outputs=[
                deblur_method,
                psf_type,
                psf_kernel_size,
                psf_sigma,
                deblur_motion_length,
                deblur_motion_angle,
                wiener_balance,
                rl_iterations,
                apply_deblur_btn,
            ],
            queue=False,
        )

        deblur_method.change(
            fn=update_deblur_parameter_visibility,
            inputs=[deblur_method, psf_type],
            outputs=[
                psf_kernel_size,
                psf_sigma,
                deblur_motion_length,
                deblur_motion_angle,
                wiener_balance,
                rl_iterations,
            ],
            queue=False,
        )

        psf_type.change(
            fn=update_deblur_parameter_visibility,
            inputs=[deblur_method, psf_type],
            outputs=[
                psf_kernel_size,
                psf_sigma,
                deblur_motion_length,
                deblur_motion_angle,
                wiener_balance,
                rl_iterations,
            ],
            queue=False,
            show_progress="hidden",
            trigger_mode="once",
        )

        apply_degradation_btn.click(
            fn=apply_degradation_if_needed,
            inputs=[
                restore_input,
                use_degradation,
                gaussian_noise_std,
                salt_pepper_amount,
                blur_type,
                blur_kernel_size,
                blur_sigma,
                degr_motion_length,
                degr_motion_angle,
            ],
            outputs=[degraded_image, degraded_state],
        )

        restoration_inputs = [
            restore_input,
            use_degradation,
            degraded_state,
            gaussian_noise_std,
            salt_pepper_amount,
            blur_type,
            blur_kernel_size,
            blur_sigma,
            degr_motion_length,
            degr_motion_angle,
            use_noise_removal,
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
            use_deblurring,
            deblur_method,
            psf_type,
            psf_kernel_size,
            psf_sigma,
            deblur_motion_length,
            deblur_motion_angle,
            wiener_balance,
            rl_iterations,
        ]

        apply_noise_btn.click(
            fn=apply_restoration,
            inputs=restoration_inputs,
            outputs=[restored_image],
        )

        apply_deblur_btn.click(
            fn=apply_restoration,
            inputs=restoration_inputs,
            outputs=[restored_image],
        )

        restore_input.change(
            fn=update_restoration_ranges,
            inputs=[restore_input],
            outputs=[
                blur_kernel_size,
                median_kernel_size,
                psf_kernel_size,
                degr_motion_length,
                deblur_motion_length,
                kuwahara_radius,
            ],
            queue=False,
        )
        restore_input.change(
            fn=reset_restoration_outputs,
            inputs=[],
            outputs=[degraded_image, restored_image, degraded_state],
            queue=False,
        )

    with gr.Tab("Documentation FR"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_fr_buttons = []
                for title in DOC_FR_TITLES:
                    btn = gr.Button(title)
                    doc_fr_buttons.append((btn, title))

            with gr.Column(scale=3):
                doc_fr_view = gr.Markdown(
                    value=load_doc_fr_section(DOC_FR_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS
                )

        for btn, title in doc_fr_buttons:
            btn.click(
                lambda t=title: load_doc_fr_section(t),
                inputs=None,
                outputs=doc_fr_view,
            )

    with gr.Tab("Documentation EN"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_en_buttons = []
                for title in DOC_EN_TITLES:
                    btn = gr.Button(title)
                    doc_en_buttons.append((btn, title))

            with gr.Column(scale=3):
                doc_en_view = gr.Markdown(
                    value=load_doc_en_section(DOC_EN_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS
                )

        for btn, title in doc_en_buttons:
            btn.click(
                lambda t=title: load_doc_en_section(t),
                inputs=None,
                outputs=doc_en_view,
            )
            
if __name__ == "__main__":
    demo.launch()