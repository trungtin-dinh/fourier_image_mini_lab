## Table of Contents

1. [The 2D Discrete Fourier Transform](#1-the-2d-discrete-fourier-transform)
2. [Frequency Filtering: General Framework](#2-frequency-filtering-general-framework)
3. [Ideal Frequency Filters](#3-ideal-frequency-filters)
4. [Gaussian Filters in the Frequency Domain](#4-gaussian-filters-in-the-frequency-domain)
5. [The 2D Butterworth Filter](#5-the-2d-butterworth-filter)
6. [Notch Reject Filter](#6-notch-reject-filter)
7. [Spatial Non-Linear Filters](#7-spatial-non-linear-filters)
8. [Fourier Phase Manipulation](#8-fourier-phase-manipulation)
9. [Degradation Model and Noise](#9-degradation-model-and-noise)
10. [Denoising: Advanced Methods](#10-denoising-advanced-methods)
11. [Deconvolution and Image Restoration](#11-deconvolution-and-image-restoration)

---

## 1. The 2D Discrete Fourier Transform

### 1.1 Definition

Let $f(x, y)$ be an image of size $H \times W$ with real-valued entries, with $x \in \{0, \ldots, W-1\}$ and $y \in \{0, \ldots, H-1\}$. The **2D Discrete Fourier Transform** (2D DFT) is defined by:

$$F(u, v) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} f(x, y)\, e^{-j2\pi\left(\frac{ux}{W} + \frac{vy}{H}\right)}$$

where $(u, v) \in \{0, \ldots, W-1\} \times \{0, \ldots, H-1\}$ are the discrete frequency coordinates. The inverse transform is:

$$f(x, y) = \frac{1}{WH} \sum_{u=0}^{W-1} \sum_{v=0}^{H-1} F(u, v)\, e^{j2\pi\left(\frac{ux}{W} + \frac{vy}{H}\right)}$$

$F(u, v)$ is a complex number: $F(u, v) = |F(u, v)|\, e^{j\angle F(u,v)}$, where $|F(u, v)|$ is the **magnitude** (spectral amplitude) and $\angle F(u, v)$ is the **spectral phase**.

### 1.2 Physical Interpretation

Each coefficient $F(u, v)$ measures the **correlation** of the image $f$ with the complex plane wave $e^{j2\pi(ux/W + vy/H)}$ — a 2D sinusoid of spatial frequency $(u/W, v/H)$ cycles per pixel in the $x$ and $y$ directions respectively. A large $|F(u, v)|$ means that the image contains strongly periodic structures at that frequency.

Low frequencies (around $(u, v) = (0, 0)$) correspond to slow variations (background, global illumination). High frequencies correspond to fine details, edges, and noise.

### 1.3 Centering and `fftshift`

By NumPy convention, $F(0, 0)$ is the DC component (the average value of the image), and high frequencies are located at the ends of the array. To visualize the spectrum with low frequencies at the center (the convention used in the app), one applies `np.fft.fftshift`: this operation performs a circular translation of $\lfloor W/2 \rfloor$ columns and $\lfloor H/2 \rfloor$ rows, placing the DC component at $(W/2, H/2)$.

In the centered spectrum, the **radial distance from the origin** of a point $(u, v)$ is:

$$d(u, v) = \sqrt{\left(u - \frac{W}{2}\right)^2 + \left(v - \frac{H}{2}\right)^2}$$

expressed in frequency pixels. It is this distance $d$ that is used to define all the frequency filtering masks in the app.

### 1.4 Magnitude and Phase Spectra

The **magnitude spectrum** displayed in the app is $\log(1 + |F_s(u,v)|)$, normalized in $[0, 1]$, where $F_s$ denotes the centered spectrum. The logarithm is essential: the DC component is typically $10^4$ times larger than the high-frequency components; without logarithmic compression, only the central spot would be visible.

The **phase spectrum** $\angle F_s(u, v) \in [-\pi, +\pi]$ is linearly remapped into $[0, 1]$ for display. Phase encodes structural spatial information: as shown by phase manipulation experiments (section 8), object recognition in an image is much better preserved by phase than by magnitude.

### 1.5 The 2D DFT as a Product of 1D DFTs

The 2D DFT is **separable**: it can be computed as a sequence of 1D DFTs, first across rows and then across columns (or the reverse):

$$F(u, v) = \sum_{y=0}^{H-1} \left[\sum_{x=0}^{W-1} f(x,y)\, e^{-j2\pi ux/W}\right] e^{-j2\pi vy/H}$$

It is this separability that allows the FFT (Fast Fourier Transform) algorithm to compute the 2D DFT very efficiently: the number of arithmetic operations is proportional to $WH\log(WH)$ — for example about 17 million operations for a $1024 \times 1024$ image — instead of approximately $W^2 H^2$, that is, more than one trillion operations for the same image with the naive definition. The FFT algorithm is therefore indispensable in practice as soon as the image exceeds a few dozen pixels per side.

---

## 2. Frequency Filtering: General Framework

### 2.1 The Convolution Theorem

The fundamental result that justifies frequency filtering is the **convolution theorem**: spatial convolution corresponds to frequency multiplication,

$$g(x, y) = (f * h)(x, y) \quad \Longleftrightarrow \quad G(u, v) = F(u, v) \cdot H(u, v)$$

where $h(x, y)$ is the impulse response of the filter and $H(u, v) = \mathcal{F}\{h\}$ its **transfer function** (or frequency response). Filtering an image therefore amounts to multiplying its spectrum pointwise by $H(u, v)$, and then computing the inverse transform.

### 2.2 Frequency Filtering Pipeline

The app systematically applies the following pipeline for each color channel $c \in \{R, G, B\}$:

$$G_c(u, v) = \mathcal{F}\{f_c\}(u, v) \cdot H(u, v)$$

$$g_c(x, y) = \text{Re}\left[\mathcal{F}^{-1}\{G_c\}(x, y)\right]$$

The real part is extracted because $f_c$ is real; the small imaginary residuals due to floating-point rounding errors are simply ignored.

**Important remark**: frequency multiplication is equivalent to **circular** (periodic) convolution in the spatial domain, not linear convolution. For an image of size $H \times W$, the result is periodized with the same period. This introduces **border artifacts** for filters with sharp transitions: structures from the right border "wrap around" into the left border. These artifacts are inherent to the DFT and can be reduced by extending the image before transformation.

### 2.3 Frequency Mask

In the app, all linear filters are defined by a **mask** $H(u, v) \in [0, 1]^{W \times H}$, a real matrix, symmetric with respect to the origin, defined in the centered frequency domain. The value $H(u, v) = 1$ means that the frequency component $(u, v)$ is fully passed; $H(u, v) = 0$ means that it is completely suppressed.

---

## 3. Ideal Frequency Filters

### 3.1 Ideal Low-Pass Filter

The ideal low-pass filter (Low-pass) with cutoff radius $D_0$ passes all frequencies whose distance from the origin is smaller than $D_0$ and rejects all the others:

$$H_{\text{LP}}(u, v) = \begin{cases} 1 & \text{if } d(u,v) \leq D_0 \\ 0 & \text{otherwise} \end{cases}$$

Its spatial impulse response $h_{\text{LP}}(x, y) = \mathcal{F}^{-1}\{H_{\text{LP}}\}$ is a **Bessel function** $J_1$ (the 2D analogue of the 1D sinc function), which oscillates around zero with decaying lobes. It is precisely this oscillatory tail that produces the **ringing** (Gibbs ringing) visible around edges in an image filtered by an ideal low-pass filter.

### 3.2 Ideal High-Pass Filter

The ideal high-pass filter is simply the complement of the low-pass filter:

$$H_{\text{HP}}(u, v) = 1 - H_{\text{LP}}(u, v) = \begin{cases} 0 & \text{if } d(u,v) \leq D_0 \\ 1 & \text{otherwise} \end{cases}$$

It suppresses low frequencies (smooth background) and preserves high frequencies (edges, details, noise). The result is an image of the "edge detection" type, gray in the background and bright at transitions.

### 3.3 Ideal Band-Pass and Band-Stop Filters

The **ideal band-pass** filter transmits an annular crown of frequencies $[D_{\text{low}}, D_{\text{high}}]$:

$$H_{\text{BP}}(u, v) = \begin{cases} 1 & \text{if } D_{\text{low}} \leq d(u,v) \leq D_{\text{high}} \\ 0 & \text{otherwise} \end{cases}$$

The **ideal band-stop** filter is its complement:

$$H_{\text{BS}}(u, v) = 1 - H_{\text{BP}}(u, v)$$

It is useful for suppressing periodic artifacts localized in a known frequency band, for example the noise of a scanline pattern.

### 3.4 Discontinuities and Gibbs Artifacts

The abrupt transition ($0 \to 1$ in a single pixel) of ideal filters produces a phenomenon analogous to the **Gibbs phenomenon** in 1D: spatial oscillations (ringing) appear near edges. These oscillations are a direct consequence of spectrum truncation: in frequency, multiplying by a disk $H_\text{LP}$ amounts to convolving the ideal impulse response (infinite) with the finite disk support, which creates side lobes. Gaussian and Butterworth filters (sections 4 and 5) are designed precisely to avoid this discontinuity.

---

## 4. Gaussian Filters in the Frequency Domain

### 4.1 The Gaussian Low-Pass Filter

The **Gaussian low-pass transfer function** with parameter $\sigma$ is:

$$H_{\text{Gauss,LP}}(u, v) = \exp\!\left(-\frac{d(u,v)^2}{2\sigma^2}\right)$$

where $d(u,v) = \sqrt{(u - W/2)^2 + (v - H/2)^2}$ is the distance from the origin in the centered spectrum.

The $-3\,\text{dB}$ cutoff frequency (gain = $1/\sqrt{2}$) is $d_{3\text{dB}} = \sigma\sqrt{\ln 2} \approx 0.832\,\sigma$.

### 4.2 Remarkable Property: Self-Transform

The Gaussian is its own Fourier transform, up to a normalization factor. More precisely, if $H_{\text{Gauss,LP}}(u,v)$ is Gaussian with variance $\sigma^2$ in the frequency domain, then $h_{\text{Gauss,LP}}(x, y) = \mathcal{F}^{-1}\{H_{\text{Gauss,LP}}\}$ is Gaussian with variance $(W H)/(4\pi^2\sigma^2)$ in the spatial domain.

This duality implies that a narrow Gaussian low-pass filter in the frequency domain (small $\sigma$) corresponds to a wide spatial convolution kernel (strong blur), and vice versa. The **product of the frequency and spatial widths** is constant — this is the expression of the **Heisenberg uncertainty principle** in 2D signal processing.

### 4.3 Gaussian High-Pass Filter

$$H_{\text{Gauss,HP}}(u, v) = 1 - H_{\text{Gauss,LP}}(u, v) = 1 - \exp\!\left(-\frac{d(u,v)^2}{2\sigma^2}\right)$$

This filter is zero for $d = 0$ (DC component suppressed) and tends to 1 for high frequencies. It produces edge enhancement without the ringing of the ideal high-pass filter, because the transition is continuous and infinitely differentiable.

### 4.4 Advantage over Ideal Filters

The Gaussian is the **only function** that has Gaussian decay both in the spatial domain and in the frequency domain. This gives it a unique property: it simultaneously minimizes spatial and frequency localization, that is, it achieves the minimum of the product $\Delta x \cdot \Delta u$ in the uncertainty principle. In practice, this means that the Gaussian filter produces **no ringing artifacts** and is entirely free of discontinuities.

---

## 5. The 2D Butterworth Filter

### 5.1 Definition

The **Butterworth low-pass transfer function** of order $n$ and cutoff radius $D_0$ is:

$$H_{\text{BW,LP}}(u, v) = \frac{1}{1 + \left(\dfrac{d(u,v)}{D_0}\right)^{2n}}$$

At the cutoff radius $d = D_0$: $H_{\text{BW,LP}} = 1/2$, that is, $-3\,\text{dB}$ (half-power gain). At the origin: $H = 1$. At infinity: $H \to 0$.

The Butterworth high-pass filter is its complement:

$$H_{\text{BW,HP}}(u, v) = 1 - H_{\text{BW,LP}}(u, v) = \frac{\left(\dfrac{d(u,v)}{D_0}\right)^{2n}}{1 + \left(\dfrac{d(u,v)}{D_0}\right)^{2n}}$$

### 5.2 Role of the Order $n$

The order $n$ controls the **slope of the transition** between the passband and the attenuated band:

- For $n = 1$: very gradual transition, comparable to a first-order RC filter.
- For $n \to \infty$: the filter converges to the ideal disk low-pass filter.

The asymptotic slope in decibels grows as $-20n\,\text{dB/decade}$ in the attenuated band, exactly as for the 1D analog Butterworth filter from which it originates. Increasing $n$ gives a sharper transition, but gradually introduces ringing, since the side lobes of the spatial impulse response increase.

### 5.3 Butterworth vs Gaussian vs Ideal Trade-Off

The three low-pass filter families represent a continuous spectrum of trade-offs:

| Property | Ideal | Butterworth ($n$ high) | Gaussian |
|---|---|---|---|
| Transition | Abrupt | Steep but continuous | Gradual |
| Spatial ringing | Strong (Gibbs) | Moderate | None |
| Monotonicity | No (oscillations) | Yes | Yes |
| Spatial localization | Poor | Good | Optimal |

The Butterworth filter is said to have a **maximally flat response** (Butterworth, 1930) in the passband: all derivatives of $|H_{\text{BW,LP}}|$ up to order $2n-1$ are zero at $d = 0$, which guarantees the absence of ripple in the passband.

---

## 6. Notch Reject Filter

### 6.1 Motivation

Periodic noise in images, such as scan interference, offset printing screen patterns, or JPEG compression artifacts, appears in the Fourier spectrum as **isolated bright spots** (peaks) symmetric with respect to the origin. A notch filter selectively suppresses these peaks without affecting the rest of the spectrum.

### 6.2 Definition

A notch filter centered at the positions $\pm(u_0, v_0)$ with radius $R$ is defined by:

$$H_{\text{notch}}(u, v) = \begin{cases} 0 & \text{if } d_1(u,v) \leq R \text{ or } d_2(u,v) \leq R \\ 1 & \text{otherwise} \end{cases}$$

where $d_1(u,v) = \sqrt{(u-u_0)^2 + (v-v_0)^2}$ and $d_2(u,v) = \sqrt{(u+u_0)^2 + (v+v_0)^2}$ are the distances to the two conjugate positions. The symmetry $\pm(u_0, v_0)$ is mandatory so that the spatial impulse response remains **real**, due to the Hermitian symmetry property of the DFT of a real signal.

### 6.3 Link with Periodic Noise

A purely sinusoidal noise component $n(x,y) = A\cos(2\pi(u_0 x/W + v_0 y/H))$ has a Fourier spectrum composed exactly of two Dirac deltas at $\pm(u_0, v_0)$. The ideal notch filter cancels these two deltas and perfectly removes the noise without affecting the rest of the signal — provided that the noise peaks do not spectrally overlap with useful signal energy.

---

## 7. Spatial Non-Linear Filters

The filters in this section operate **directly in the spatial domain** and are intrinsically non-linear: they cannot be expressed as a convolution with a fixed kernel, and therefore have no transfer function $H(u,v)$ in the LTI sense.

### 7.1 Median Filter

For each pixel $(x_0, y_0)$, the median filter of size $k \times k$ replaces the value by the **median** of the $k^2$ values in the neighborhood:

$$g(x_0, y_0) = \text{median}\!\left\{\, f(x_0 + i,\, y_0 + j) : (i,j) \in \left[-\lfloor k/2 \rfloor, \lfloor k/2 \rfloor\right]^2 \right\}$$

The median is an **order statistic**: it depends on the central value among the sorted values, which makes it robust to outliers. An impulse-noise pixel in the neighborhood is an outlier among $k^2 - 1$ normal values; it does not influence the median as soon as $k \geq 3$. By contrast, an arithmetic mean would propagate this outlier to the output pixel.

**Edge preservation**: if at most $\lfloor k^2/2 \rfloor$ pixels in the neighborhood cross an edge, the median remains on the same side as the central pixel, preserving the spatial transition. An averaging filter of the same size would blur the edge over $k$ pixels.

### 7.2 Minimum Filter (Erosion)

$$g(x_0, y_0) = \min_{(i,j) \in \mathcal{W}} f(x_0 + i,\, y_0 + j)$$

The minimum filter is the morphological operation of **erosion** with a square structuring element of size $k \times k$. It shrinks bright regions and expands dark regions. In practice, it darkens the image and widens dark areas. It is notably used to remove isolated bright pixels ("salt" noise) and for morphological shape analysis.

### 7.3 Maximum Filter (Dilation)

$$g(x_0, y_0) = \max_{(i,j) \in \mathcal{W}} f(x_0 + i,\, y_0 + j)$$

The maximum filter is the morphological operation of **dilation**. It brightens the image and widens bright areas. It removes isolated dark pixels ("pepper" noise). The composition dilation then erosion gives the **morphological closing**; erosion then dilation gives the **opening**, both of which are fundamental operations in mathematical morphology.

### 7.4 Bilateral Filter

The bilateral filter (Tomasi & Manduchi, 1998) is a spatial smoothing filter that preserves edges by weighting each neighbor not only by its **spatial distance** but also by its **radiometric similarity** (difference in value) to the central pixel:

$$g(x_0, y_0) = \frac{1}{W_p} \sum_{(i,j) \in \mathcal{W}} f(x_0+i,\, y_0+j)\, k_s(i, j)\, k_r\!\left(f(x_0+i,\, y_0+j) - f(x_0, y_0)\right)$$

where the **normalization constant** is $W_p = \sum_{(i,j)} k_s(i,j)\, k_r(f(x_0+i,y_0+j) - f(x_0,y_0))$, and the two kernels are Gaussian:

$$k_s(i, j) = \exp\!\left(-\frac{i^2 + j^2}{2\sigma_s^2}\right), \qquad k_r(\Delta) = \exp\!\left(-\frac{\Delta^2}{2\sigma_r^2}\right)$$

The parameter $\sigma_s$ (spatial sigma) controls the spatial extent of the filter. The parameter $\sigma_r$ (radiometric sigma or color sigma) controls the intensity selectivity: a pixel whose value differs by more than $2\sigma_r$ from the central pixel receives an almost zero weight, which prevents smoothing across edges.

**Key property**: in homogeneous regions, $k_r \approx 1$ for all neighbors, and the bilateral filter reduces to an ordinary Gaussian filter. Near an edge, pixels on the other side of the edge have $k_r \approx 0$ and do not participate in the average, so the edge is preserved.

---

## 8. Fourier Phase Manipulation

### 8.1 Amplitude-Phase Decomposition

Any spectrum $F(u, v) \in \mathbb{C}$ can be written in polar coordinates as:

$$F(u, v) = A(u, v)\, e^{j\phi(u, v)}$$

where $A(u, v) = |F(u, v)| \geq 0$ is the **spectral amplitude** and $\phi(u, v) = \angle F(u, v) \in (-\pi, +\pi]$ is the **spectral phase**. Phase manipulation experiments reveal the respective roles of these two components in visual perception.

### 8.2 Phase-Only Reconstruction

$$G_{\text{phase}}(u, v) = e^{j\phi(u, v)} \qquad \text{(amplitude uniformly replaced by 1)}$$

$$g_{\text{phase}}(x, y) = \mathcal{F}^{-1}\{G_{\text{phase}}\}(x, y)$$

All amplitude information is removed; only the phase structure is retained. The reconstruction $g_{\text{phase}}$ is normalized for display. Empirical result: the **edges and geometric structure** of objects remain clearly recognizable, although gray levels and textures are altered. This demonstrates that **phase encodes structural information**, namely the positions of discontinuities and alignments.

### 8.3 Amplitude-Only Reconstruction

$$G_{\text{amp}}(u, v) = A(u, v) \qquad \text{(phase uniformly set to zero)}$$

$$g_{\text{amp}}(x, y) = \mathcal{F}^{-1}\{A(u,v)\}(x, y)$$

By canceling all phase information, spatial structure is lost. Since $A(u,v) \geq 0$ is real, its inverse transform has Hermitian symmetry and exhibits characteristic artifacts, namely concentration of energy around the spatial center. The image becomes unrecognizable. This result, symmetric to 8.2, confirms that amplitude contains the **spectral statistics** (energy distribution across frequencies), but not the spatial organization.

### 8.4 Random Phase

Phase is replaced by a uniformly random field $\phi_{\text{rand}}(u, v) \sim \mathcal{U}(-\pi, +\pi)$, independently for each coefficient:

$$G_{\text{rand}}(u, v) = A(u, v)\, e^{j\phi_{\text{rand}}(u, v)}$$

The original amplitude spectrum is preserved, so the **statistical distribution** of gray levels is preserved, with the same histogram, but all spatial structure is destroyed. The reconstructed image looks like colored noise whose power spectrum $|A(u,v)|^2$ matches the original one.

### 8.5 Phase Quantization

Phase is quantized over $L$ uniformly distributed levels in $(-\pi, +\pi]$:

$$\phi_{\text{quant}}(u, v) = -\pi + \Delta\phi \cdot \text{round}\!\left(\frac{\phi(u,v) + \pi}{\Delta\phi}\right), \qquad \Delta\phi = \frac{2\pi}{L}$$

For $L = 2$, only two levels $\{-\pi, 0\}$ are possible. For $L = 8$ or more, the visual structure is almost entirely preserved despite the strong quantization. This indicates that phase does not need arbitrarily high precision to remain informative — a few bits of angular resolution are enough to recognize a scene.

### 8.6 Phase Noise

Gaussian noise is added to the phase:

$$\phi_{\text{noisy}}(u, v) = \phi(u, v) + \sigma_\phi\, \eta(u, v), \qquad \eta(u, v) \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$$

For $\sigma_\phi$ small compared with $\pi$, the image remains recognizable. For $\sigma_\phi \gg \pi$, the phase becomes essentially random and the image degenerates as in random-phase reconstruction, as described in section 8.4.

### 8.7 Linear Phase Ramp (Spatial Shift)

The shift property of the DFT states that multiplying $F(u,v)$ by $e^{-j2\pi(u\Delta x/W + v\Delta y/H)}$ amounts to translating $f(x,y)$ by $(\Delta x, \Delta y)$ pixels in the spatial domain:

$$\mathcal{F}^{-1}\!\left\{F(u,v)\, e^{-j2\pi\left(\frac{u\,\Delta x}{W} + \frac{v\,\Delta y}{H}\right)}\right\}(x, y) = f(x - \Delta x,\, y - \Delta y)$$

Adding a linear ramp to the phase, $\phi_{\text{ramp}}(u,v) = \phi(u,v) - 2\pi(u\,\Delta x/W + v\,\Delta y/H)$, therefore produces a **circular translation** of the image, with no modification of the spectral amplitude. The translated image is "wrapped", so that what exits one border reappears on the opposite border. This is the most direct demonstration that phase encodes the **spatial position** of information.

---

## 9. Degradation Model and Noise

### 9.1 General Linear Degradation Model

The standard image degradation model used in restoration is:

$$f_{\text{deg}}(x, y) = (h * f)(x, y) + n(x, y)$$

where $f(x, y)$ is the original image, $h(x, y)$ is the **impulse response of the degradation system** (PSF — Point Spread Function), $*$ denotes convolution, and $n(x, y)$ is additive noise. In the Fourier domain:

$$F_{\text{deg}}(u, v) = H_{\text{PSF}}(u, v) \cdot F(u, v) + N(u, v)$$

The goal of restoration is to estimate $F(u,v)$ from $F_{\text{deg}}(u,v)$, knowing or estimating $H_{\text{PSF}}$ and the statistics of $N$.

### 9.2 Additive Gaussian Noise

Additive Gaussian noise models sensor thermal noise:

$$f_{\text{noisy}}(x, y) = f(x, y) + \eta(x, y), \qquad \eta(x, y) \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_n^2)$$

In practice, with images normalized in $[0,1]$, the parameter $\sigma_n$ controls the noise level: $\sigma_n = 0.01$ is almost imperceptible, while $\sigma_n = 0.1$ is heavily noisy.

The power spectral density of white Gaussian noise is **flat**: the average power of each frequency coefficient $N(u,v)$ is $W H \sigma_n^2$, regardless of the frequency $(u,v)$. Noise is uniformly distributed across the entire spectrum, which is precisely what makes it difficult to remove without degrading the signal.

### 9.3 Salt-and-Pepper Noise

Salt-and-pepper noise models defective pixels or transmission errors: a fraction $\rho$ of pixels is replaced by either the maximum value (salt, $= 1$) or the minimum value (pepper, $= 0$), independently:

$$f_{\text{sp}}(x, y) = \begin{cases} 1 & \text{with probability } \rho/2 \\ 0 & \text{with probability } \rho/2 \\ f(x, y) & \text{with probability } 1 - \rho \end{cases}$$

Unlike Gaussian noise, corrupted pixels are **extreme outliers** of known value. The median filter, as described in section 7.1, is the optimal denoiser for this type of noise.

### 9.4 Point Spread Function (PSF): Gaussian Blur

The Gaussian PSF models defocus blur or optical diffusion blur:

$$h_{\text{Gauss}}(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

In practice, it is discretized on a $k \times k$ grid with odd $k$ and normalized so that $\sum_{x,y} h(x,y) = 1$, which ensures energy conservation. Convolution with $h_{\text{Gauss}}$ is applied by `fftconvolve` in `"same"` mode, which corresponds to linear convolution truncated to the size of the input image.

### 9.5 Point Spread Function: Motion Blur

The motion PSF models motion blur caused by the movement of the camera or the subject during exposure. It is approximated by a line segment of length $L$ and angle $\theta$:

$$h_{\text{motion}}(x, y) = \frac{1}{L}\, \mathbf{1}\!\left[(x, y) \text{ on the segment of length } L \text{ and angle } \theta\right]$$

In the Fourier domain, this PSF has a sinc-shaped form oriented perpendicular to the direction of motion, with regularly spaced zeros that make deconvolution difficult, since noise is amplified at the frequencies canceled by $H_{\text{PSF}}$.

---

## 10. Denoising: Advanced Methods

### 10.1 Total Variation (TV)

**Total variation regularization** (Rudin, Osher & Fatemi, 1992) estimates the restored image $g$ by minimizing a variational problem that balances data fidelity and solution regularity:

$$\hat{g} = \arg\min_{g} \left\{ \frac{1}{2}\|g - f_{\text{noisy}}\|_2^2 + \lambda \cdot \text{TV}(g) \right\}$$

The **isotropic total variation** is defined by:

$$\text{TV}(g) = \sum_{x,y} \|\nabla g(x,y)\|_2 = \sum_{x,y} \sqrt{\left(\frac{\partial g}{\partial x}\right)^2 + \left(\frac{\partial g}{\partial y}\right)^2}$$

The first term $\|g - f_{\text{noisy}}\|_2^2$ is the **fidelity** term: it penalizes solutions that deviate from the noisy image. The second term $\text{TV}(g)$ is the **regularization** term: it penalizes large spatial variations. The parameter $\lambda$, called `weight` in the app, controls the trade-off.

The fundamental property of TV regularization is the **preservation of sharp edges**: unlike regularization by the sum of squared gradients, which penalizes $\sum_{x,y}\|\nabla g\|^2$ and produces progressive Gaussian smoothing, regularization by the sum of the **norms** of the gradient, that is, the term $\text{TV}(g)$, favors piecewise-constant solutions with sharp transitions, because canceling the gradient in a region costs little, whereas a large gradient value is strongly penalized. The optimization algorithm is Chambolle's method (2004), a dual subgradient descent method.

### 10.2 Non-Local Means (NLM)

The **Non-Local Means** method (Buades, Coll & Morel, 2005) generalizes the bilateral filter by comparing not individual pixels but **patches** centered on each pixel:

$$g(x_0, y_0) = \frac{1}{Z(x_0, y_0)} \sum_{(x, y) \in \Omega} f(x, y)\, \exp\!\left(-\frac{\|P(x_0, y_0) - P(x, y)\|_{2,a}^2}{h^2}\right)$$

where $P(x, y)$ denotes the patch of size $p \times p$ centered at $(x,y)$, $\|\cdot\|_{2,a}^2$ is the $\ell^2$ norm weighted by a Gaussian kernel with parameter $a$, $h$ is the filtering parameter, and $Z(x_0, y_0) = \sum_{(x,y)} \exp(-\|\ldots\|^2/h^2)$ is the normalization constant.

The patch similarity measures the **resemblance of local texture**: two pixels having similar neighborhoods, with the same texture and orientation, receive a high weight and are averaged together, even if they are spatially far apart in the image. This is the strength of NLM: it exploits the **non-local redundancy** of the image, namely the fact that natural textures repeat.

The key parameters are the patch size $p$, the maximum search distance, and $h$, which controls the selectivity of the patch comparison. The naive implementation compares each pixel with all other pixels in the image, which results in a computational cost proportional to $W^2 H^2 p^2$, prohibitive for large images. The `fast_mode=True` version uses an approximation based on **patch integrals**, computing distances between patches through 2D cumulative sums in a way analogous to the integral image described for Kuwahara, which reduces the cost to $W H p^2$, proportional to the number of pixels and independent of the search window.

### 10.3 Wavelet Denoising

**Wavelets** provide a **multi-resolution** representation of the image: the 2D wavelet transform decomposes $f$ into a series of detail subbands, horizontal, vertical, and diagonal, at different scales, plus a low-frequency approximation subband.

For a noisy image $f = s + \eta$ where $\eta \sim \mathcal{N}(0, \sigma_n^2)$:

- The **signal wavelet coefficients** of $s$ are concentrated, that is, few in number and of large magnitude, for natural images.
- The **noise wavelet coefficients** of $\eta$ are uniformly distributed across all subbands with variance $\sigma_n^2$.

This **sparsity** of the signal coefficients is the fundamental principle of the wavelet approach: **thresholding** the coefficients, keeping the large ones and setting the small ones to zero, selectively removes noise while preserving the signal.

The app uses the **BayesShrink** method (Chang, Yu & Vetterli, 2000), which adaptively estimates the optimal threshold for each subband from the local variance of the coefficients:

$$\hat{\sigma}_s = \sqrt{\max\!\left(0, \hat{\sigma}_y^2 - \sigma_n^2\right)}, \qquad \tau_k = \frac{\sigma_n^2}{\hat{\sigma}_{s,k}}$$

where $\hat{\sigma}_y^2$ is the variance of the observed coefficients in subband $k$ and $\hat{\sigma}_{s,k}$ is the estimate of the signal variance in that subband. This threshold minimizes the Bayesian mean squared error under a Laplace prior on the signal coefficients.

### 10.4 Kuwahara Filter

The Kuwahara filter (Kuwahara et al., 1976) is an adaptive smoothing filter that preserves edges by selecting the most homogeneous neighborhood quadrant. For each pixel $(x_0, y_0)$, the neighborhood of radius $r$ is divided into four quadrants that slightly overlap:

$$Q_k = \{(x_0 + i,\, y_0 + j) : (i, j) \in \text{quadrant } k\}, \quad k = 1, 2, 3, 4$$

The **variance** $\sigma_k^2$ and the **mean** $\mu_k$ of the intensity, here of the grayscale image used as a guide, are computed in each quadrant. The output is the mean of the quadrant with minimum variance:

$$g(x_0, y_0) = \mu_{k^*}, \qquad k^* = \arg\min_k \sigma_k^2$$

The integral image, also called summed area table, is used to compute $\mu_k$ and $\sigma_k^2$ in a **constant** number of operations per pixel, regardless of the radius $r$. Indeed, the sum over any rectangle in an array can be obtained with exactly 4 additions or subtractions on the cumulative sum table, independently of the rectangle size. The total complexity is therefore proportional to the number of pixels $W \times H$, and does **not** increase with $r$, which is a crucial optimization for large radii, since without this trick the cost would be proportional to $W \times H \times r^2$.

**Key property**: on the homogeneous side of an edge, the variance is low and that quadrant is selected; on the heterogeneous side, which crosses the edge, the variance is high and that quadrant is rejected. The result is strong smoothing in homogeneous regions and preservation of edges.

---

## 11. Deconvolution and Image Restoration

### 11.1 The Deconvolution Problem

Deconvolution is the inverse problem of blurring. Given the blurred image $f_{\text{deg}} = h * f + n$ and the PSF $h$, one seeks to estimate $f$. In the Fourier domain, the naive solution, namely inverse filtering, would be:

$$\hat{F}(u, v) = \frac{F_{\text{deg}}(u, v)}{H_{\text{PSF}}(u, v)}$$

This inverse filter is **unstable**: at frequencies where $H_{\text{PSF}}(u,v) \approx 0$, corresponding to zeros of the PSF, the term $N(u,v)/H_{\text{PSF}}(u,v)$ diverges, and the noise is amplified catastrophically. Regularization is therefore indispensable.

### 11.2 Wiener Filter for Deconvolution

The Wiener filter solves the deconvolution problem by seeking the linear filter $W(u,v)$ that minimizes the mean squared error between the estimate $\hat{F}$ and the true $F$:

$$\hat{F}(u, v) = W(u, v) \cdot F_{\text{deg}}(u, v)$$

$$W(u, v) = \frac{H_{\text{PSF}}^*(u, v)}{|H_{\text{PSF}}(u, v)|^2 + \underbrace{S_n(u,v)/S_f(u,v)}_{\text{NSR}(u,v)}}$$

where $H_{\text{PSF}}^*$ is the complex conjugate of $H_{\text{PSF}}$, $S_n(u,v) = \mathbb{E}[|N(u,v)|^2]$ is the noise power spectrum, and $S_f(u,v) = \mathbb{E}[|F(u,v)|^2]$ is the signal power spectrum. The ratio $\text{NSR}(u,v) = S_n/S_f$ is the **local noise-to-signal ratio**.

In the app, NSR is assumed constant: $\text{NSR}(u,v) = K$, the **Wiener balance** parameter $K$. The Wiener filter then becomes:

$$W(u, v) = \frac{H_{\text{PSF}}^*(u, v)}{|H_{\text{PSF}}(u, v)|^2 + K}$$

For $K \to 0$, the Wiener filter converges to the pure inverse filter, unstable and highly noise-sensitive. For large $K$, the filter attenuates all frequencies and produces a blurred but non-amplified image. The optimal value of $K$ balances blur removal against noise amplification.

### 11.3 Unsupervised Wiener Filter

The unsupervised version, Unsupervised Wiener (Orieux, Giovannelli & Rodet, 2010), jointly estimates the restored image and the spectral hyperparameters, namely the power spectra $S_f$ and $S_n$, directly from the data, without prior knowledge of the noise level. The estimation is carried out by an MCMC algorithm, Markov Chain Monte Carlo, alternating between sampling the restored image and updating the hyperparameters. This method is more robust than supervised Wiener in cases where the noise level is unknown.

### 11.4 Richardson-Lucy (Poisson Maximum-Likelihood Deconvolution)

The **Richardson-Lucy** algorithm (Richardson, 1972; Lucy, 1974) is an iterative deconvolution algorithm derived from the **Poisson noise** model, which is appropriate for photon noise. Under the assumption that each pixel $f_{\text{deg}}(x,y)$ is a sample from a Poisson distribution with parameter $(h * f)(x,y)$, the log-likelihood to maximize is:

$$\ell(f) = \sum_{x,y} \left[ f_{\text{deg}}(x,y) \log\!\left((h * f)(x,y)\right) - (h * f)(x,y) \right]$$

The associated EM algorithm, Expectation-Maximization, yields the Richardson-Lucy iterative update rule:

$$f^{(t+1)}(x, y) = f^{(t)}(x, y) \cdot \left(h(-\cdot,-\cdot) * \frac{f_{\text{deg}}}{h * f^{(t)}}\right)(x, y)$$

where $h(-x,-y)$ is the flipped PSF, that is, correlation. Writing $\tilde{h}(x,y) = h(-x,-y)$, the iteration becomes:

$$f^{(t+1)} = f^{(t)} \cdot \left[\tilde{h} * \frac{f_{\text{deg}}}{h * f^{(t)}}\right]$$

Each iteration is a pointwise multiplication by a corrective term. The algorithm converges toward the Poisson maximum-likelihood solution, which is the maximum a posteriori estimator under a uniform prior on $f$.

**Behavior with respect to the number of iterations**: for a small number of iterations, the algorithm acts like a partial low-pass filter. As the number of iterations increases, the resolution improves but the noise is progressively amplified, corresponding to overfitting the noise fluctuations. There exists an **optimal number of iterations** that minimizes the quadratic error, beyond which the image degrades.