---
title: Fourier Image Mini Lab
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive Fourier filtering, denoising and deblurring app
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Fourier Image Mini Lab

This repository contains an interactive image processing mini lab focused on Fourier analysis, frequency-domain filtering, image degradation, denoising, deblurring, and phase manipulation.

The app is designed as an educational and portfolio demo for signal and image processing. It allows the user to experiment with classical Fourier-domain operations and spatial restoration methods while observing the corresponding visual effects directly.

A Streamlit deployment is available here:

https://fourier-image-minilab.streamlit.app/

## Main features

- Load an image or use the default `skimage.data.coffee()` image.
- Display the input image, magnitude spectrum, and phase spectrum.
- Apply Fourier-domain linear filters:
  - low-pass,
  - high-pass,
  - ideal band-pass,
  - ideal band-stop,
  - Gaussian low-pass,
  - Gaussian high-pass,
  - Butterworth low-pass,
  - Butterworth high-pass,
  - notch reject.
- Apply spatial non-linear filters:
  - median filter,
  - minimum filter,
  - maximum filter,
  - bilateral filter.
- Manipulate the Fourier phase:
  - phase-only reconstruction,
  - magnitude-only reconstruction,
  - random phase reconstruction,
  - phase quantization,
  - phase noise,
  - linear phase ramp.
- Simulate image degradation:
  - additive Gaussian noise,
  - salt-and-pepper noise,
  - Gaussian blur,
  - motion blur.
- Apply denoising methods:
  - median filtering,
  - Gaussian filtering,
  - total variation denoising,
  - bilateral filtering,
  - Kuwahara filtering,
  - non-local means,
  - wavelet denoising.
- Apply deblurring methods:
  - Wiener deconvolution,
  - Richardson-Lucy deconvolution,
  - unsupervised Wiener deconvolution.
- Read the English and French documentation tabs.

## Method overview

The app is organised around four complementary image processing themes:

```text
Input image
    |
    +-- Fourier analysis and frequency filtering
    |
    +-- Spatial non-linear filtering
    |
    +-- Phase manipulation experiments
    |
    +-- Degradation, denoising, and deblurring
```

The goal is not only to process an image, but also to make the role of each operation visually interpretable.

## Fourier analysis

The app computes the 2D Discrete Fourier Transform of the image.

The magnitude spectrum shows how the image energy is distributed across spatial frequencies. Low frequencies correspond to smooth structures and illumination. High frequencies correspond to edges, fine details, texture, and noise.

The phase spectrum stores spatial alignment information. The phase manipulation tab shows that the geometric structure of an image is often much more strongly tied to phase than to magnitude.

## Frequency-domain filtering

The app applies frequency filters by multiplying the centered Fourier spectrum by a mask.

The implemented filters cover several classical families.

Ideal filters have abrupt transitions and therefore demonstrate ringing artifacts. Gaussian filters have smooth transitions and avoid ringing. Butterworth filters provide an intermediate compromise controlled by the filter order. The notch reject filter selectively suppresses localized periodic frequency components.

These filters are useful for demonstrating the link between spectrum shaping and spatial-domain visual effects.

## Spatial non-linear filtering

The app also includes spatial filters that do not correspond to a simple linear convolution.

The median filter is useful for impulse noise. The minimum and maximum filters correspond to erosion-like and dilation-like behavior. The bilateral filter smooths homogeneous regions while preserving edges by combining spatial proximity and intensity similarity.

These methods make it possible to compare linear Fourier filtering with non-linear spatial processing.

## Fourier phase manipulation

The app includes several phase experiments:

- phase-only reconstruction,
- magnitude-only reconstruction,
- random phase reconstruction,
- phase quantization,
- phase noise,
- linear phase ramp.

These experiments illustrate that the Fourier phase carries essential spatial information.

A phase-only reconstruction often preserves recognizable structure, while a magnitude-only reconstruction loses most object geometry. A linear phase ramp produces a circular spatial shift, directly showing the link between phase and position.

## Degradation model

The restoration part of the app follows the classical image degradation model:

```text
degraded image = blur(original image) + noise
```

The blur can be generated using either a Gaussian point spread function or a motion-blur point spread function. Noise can be added as Gaussian noise or salt-and-pepper noise.

This makes it possible to create a controlled degraded image before applying denoising or deblurring methods.

## Denoising

The app includes several denoising approaches.

Median filtering is particularly adapted to salt-and-pepper noise. Gaussian filtering smooths noise but also blurs edges. Total variation denoising preserves sharp transitions better than simple smoothing. Bilateral filtering preserves edges by using intensity-aware weights. Non-local means exploits patch similarity. Wavelet denoising works by attenuating noise in a multiscale transform domain.

The goal is to compare the strengths and limitations of each method on the same degraded image.

## Deblurring

The app implements three deconvolution methods.

Wiener deconvolution uses a regularised inverse filter. Richardson-Lucy deconvolution is an iterative restoration method often used for images degraded by a known point spread function. Unsupervised Wiener deconvolution estimates restoration parameters automatically.

These methods are useful for understanding why deblurring is an ill-posed problem: when a blur suppresses frequency components, inverse filtering can strongly amplify noise.

## Repository structure

```text
.
├── app.py                 # Gradio / Hugging Face Space entry point
├── app_sl.py              # Streamlit version of the app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # License file
└── README.md              # Repository and Hugging Face Space description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/fourier_image_mini_lab.git
cd fourier_image_mini_lab
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The repository requirements include:

```text
gradio
numpy
pillow
scipy
scikit-image
PyWavelets
```

If you want to run the Streamlit version locally and Streamlit is not already installed, install it as well:

```bash
pip install streamlit
```

## Run the Gradio app

```bash
python app.py
```

The local interface will usually be available at:

```text
http://127.0.0.1:7860
```

## Run the Streamlit app

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Hugging Face Space notes

The YAML block at the top of this README is used by Hugging Face Spaces.

The current metadata launches the Gradio version:

```yaml
sdk: gradio
app_file: app.py
```

If you want Hugging Face to launch the Streamlit version instead, update the metadata to:

```yaml
sdk: streamlit
app_file: app_sl.py
```

In that case, make sure `streamlit` is included in `requirements.txt`.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation.
- `documentation_fr.md` for the French documentation.

These files explain the 2D Discrete Fourier Transform, frequency filtering, ideal filters, Gaussian filters, Butterworth filters, notch reject filters, spatial non-linear filtering, Fourier phase manipulation, degradation models, denoising, deconvolution, and image restoration.

## Notes and limitations

This app is intended as an educational mini lab.

Some operations, especially non-local means, wavelet denoising, Richardson-Lucy deconvolution, and unsupervised Wiener restoration, may take longer on large images. The default image is therefore a moderate-size sample image suitable for online execution.

Restoration results depend strongly on the chosen degradation model and parameters. In particular, deconvolution is sensitive to noise and to mismatch between the assumed point spread function and the actual blur.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image, and computer vision mini apps.
