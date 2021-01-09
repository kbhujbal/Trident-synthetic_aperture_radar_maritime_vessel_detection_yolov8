"""
SAR Image Preprocessing Module - Speckle Filtering

WHY SPECKLE FILTERING IS CRITICAL FOR SAR IMAGERY
=================================================

Synthetic Aperture Radar (SAR) images suffer from a unique noise phenomenon called
"speckle noise". Unlike optical sensors that capture reflected light, SAR sends
microwave pulses and records the backscattered signal. The coherent nature of radar
causes constructive and destructive interference patterns, resulting in a granular,
"salt-and-pepper" appearance.

THE PROBLEM WITH SPECKLE:
-------------------------
1. Ships appear as bright clusters mixed with noise
2. Standard CNN detectors struggle to distinguish ships from noise
3. False positives increase dramatically
4. Small vessels become invisible

THE SOLUTION - LEE FILTER:
--------------------------
The Lee filter is a multiplicative noise filter specifically designed for SAR images.
It works by:
1. Estimating local mean and variance in a sliding window
2. Weighting pixels based on local statistics
3. Preserving edges while smoothing homogeneous regions

This is the "secret sauce" that transforms noisy SAR imagery into clean inputs
suitable for deep learning models.

MATHEMATICAL FOUNDATION:
------------------------
The Lee filter assumes multiplicative noise model: I = R * N
Where:
- I = observed (noisy) image
- R = true reflectance (what we want)
- N = speckle noise (multiplicative, not additive)

The filter estimates R using local statistics:
R_hat = mean(I) + W * (I - mean(I))

Where W is the adaptive weight:
W = var(I) / (var(I) + var_noise)

This preserves edges (high local variance) while smoothing flat regions.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Union
from pathlib import Path


def apply_lee_filter(
    image: np.ndarray,
    window_size: int = 7,
    noise_variance: Optional[float] = None
) -> np.ndarray:
    """
    Apply Lee speckle filter to a SAR image.

    The Lee filter is the gold standard for SAR despeckling because it:
    1. Assumes multiplicative noise (correct for SAR)
    2. Adapts to local image statistics
    3. Preserves edges while smoothing flat regions

    Args:
        image: Input SAR image (grayscale or BGR)
        window_size: Size of the filtering window (must be odd). Larger windows
                     provide more smoothing but may blur edges. 7x7 is standard.
        noise_variance: Estimated noise variance. If None, it's computed from
                        the image using a homogeneous region assumption.

    Returns:
        Filtered image with reduced speckle noise

    Example:
        >>> sar_image = cv2.imread('sar_ship.jpg', cv2.IMREAD_GRAYSCALE)
        >>> filtered = apply_lee_filter(sar_image, window_size=7)
        >>> cv2.imwrite('sar_ship_filtered.jpg', filtered)
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    # Convert to float for precision
    if image.dtype != np.float64:
        img_float = image.astype(np.float64)
    else:
        img_float = image.copy()

    # Handle color images by processing each channel
    if len(img_float.shape) == 3:
        channels = cv2.split(img_float)
        filtered_channels = [
            _lee_filter_single_channel(ch, window_size, noise_variance)
            for ch in channels
        ]
        return cv2.merge(filtered_channels).astype(image.dtype)

    return _lee_filter_single_channel(img_float, window_size, noise_variance).astype(image.dtype)


def _lee_filter_single_channel(
    image: np.ndarray,
    window_size: int,
    noise_variance: Optional[float] = None
) -> np.ndarray:
    """
    Apply Lee filter to a single channel image.

    This is the core Lee filter implementation following the original paper:
    "Speckle analysis and smoothing of synthetic aperture radar images"
    by Jong-Sen Lee, 1981.
    """
    # Avoid division by zero
    img = np.clip(image, 1e-10, None)

    # Local mean using uniform filter
    local_mean = ndimage.uniform_filter(img, size=window_size, mode='reflect')

    # Local squared mean for variance calculation
    local_sq_mean = ndimage.uniform_filter(img ** 2, size=window_size, mode='reflect')

    # Local variance: E[X^2] - E[X]^2
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # Estimate noise variance if not provided
    # Using the coefficient of variation method
    if noise_variance is None:
        # For multiplicative noise in SAR, noise variance relates to the
        # Equivalent Number of Looks (ENL)
        # We estimate it from homogeneous regions (low local variance)
        homogeneous_mask = local_var < np.percentile(local_var, 25)
        if np.any(homogeneous_mask):
            noise_variance = np.mean(local_var[homogeneous_mask])
        else:
            # Fallback: assume moderate noise
            noise_variance = np.mean(local_var) * 0.5

    # Ensure non-zero noise variance
    noise_variance = max(noise_variance, 1e-10)

    # Lee filter weight calculation
    # W = var_signal / (var_signal + var_noise)
    # Where var_signal = var_total - var_noise
    var_signal = np.maximum(local_var - noise_variance, 0)
    weight = var_signal / (var_signal + noise_variance)

    # Apply the Lee filter formula
    # R = mean + W * (I - mean)
    filtered = local_mean + weight * (img - local_mean)

    return filtered


def apply_enhanced_lee_filter(
    image: np.ndarray,
    window_size: int = 7,
    damping_factor: float = 1.0
) -> np.ndarray:
    """
    Apply Enhanced Lee filter with additional damping control.

    The Enhanced Lee filter extends the standard Lee filter with:
    1. Better edge preservation through exponential damping
    2. More aggressive noise removal in homogeneous regions
    3. Adaptive threshold for edge detection

    Args:
        image: Input SAR image
        window_size: Size of the filtering window
        damping_factor: Controls smoothing strength (0.5-2.0 typical)
                        Lower = more smoothing, Higher = more preservation

    Returns:
        Enhanced filtered image
    """
    if window_size % 2 == 0:
        window_size += 1

    img_float = image.astype(np.float64)

    if len(img_float.shape) == 3:
        channels = cv2.split(img_float)
        filtered_channels = [
            _enhanced_lee_single_channel(ch, window_size, damping_factor)
            for ch in channels
        ]
        return cv2.merge(filtered_channels).astype(image.dtype)

    return _enhanced_lee_single_channel(img_float, window_size, damping_factor).astype(image.dtype)


def _enhanced_lee_single_channel(
    image: np.ndarray,
    window_size: int,
    damping_factor: float
) -> np.ndarray:
    """Enhanced Lee filter for single channel."""
    img = np.clip(image, 1e-10, None)

    # Compute local statistics
    local_mean = ndimage.uniform_filter(img, size=window_size, mode='reflect')
    local_sq_mean = ndimage.uniform_filter(img ** 2, size=window_size, mode='reflect')
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # Coefficient of variation (CV) = sigma / mean
    # High CV indicates edges, low CV indicates homogeneous regions
    cv = np.sqrt(local_var) / (local_mean + 1e-10)

    # Estimate noise CV from homogeneous regions
    noise_cv = np.percentile(cv, 10)

    # Enhanced weighting using exponential damping
    cv_ratio = cv / (noise_cv + 1e-10)
    weight = np.exp(-damping_factor * cv_ratio)
    weight = np.clip(weight, 0, 1)

    # Apply weighted combination
    filtered = weight * local_mean + (1 - weight) * img

    return filtered


class SARPreprocessor:
    """
    Complete SAR image preprocessing pipeline.

    This class encapsulates all preprocessing steps needed before feeding
    SAR images to a detection model. It ensures consistent preprocessing
    across training and inference.

    WHY USE A CLASS?
    ----------------
    1. Encapsulates preprocessing parameters
    2. Ensures training and inference use identical preprocessing
    3. Supports method chaining for pipeline construction
    4. Allows easy extension with additional filters
    """

    def __init__(
        self,
        lee_window_size: int = 7,
        lee_noise_variance: Optional[float] = None,
        use_enhanced_lee: bool = False,
        enhance_contrast: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False
    ):
        """
        Initialize the SAR preprocessor.

        Args:
            lee_window_size: Window size for Lee filter (7 is standard)
            lee_noise_variance: Noise variance estimate (auto if None)
            use_enhanced_lee: Use enhanced Lee filter for stronger despeckling
            enhance_contrast: Apply CLAHE contrast enhancement after filtering
            target_size: Resize images to (width, height) if specified
            normalize: Normalize pixel values to [0, 1]
        """
        self.lee_window_size = lee_window_size
        self.lee_noise_variance = lee_noise_variance
        self.use_enhanced_lee = use_enhanced_lee
        self.enhance_contrast = enhance_contrast
        self.target_size = target_size
        self.normalize = normalize

        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to an image.

        Pipeline:
        1. Speckle filtering (Lee filter) - Remove radar noise
        2. Contrast enhancement (CLAHE) - Improve ship visibility
        3. Resizing (if configured) - Standardize input size
        4. Normalization (if configured) - Scale to [0, 1]

        Args:
            image: Input SAR image (BGR or grayscale)

        Returns:
            Preprocessed image ready for detection
        """
        result = image.copy()

        # Step 1: Speckle filtering - THE CRITICAL STEP
        if self.use_enhanced_lee:
            result = apply_enhanced_lee_filter(
                result,
                window_size=self.lee_window_size
            )
        else:
            result = apply_lee_filter(
                result,
                window_size=self.lee_window_size,
                noise_variance=self.lee_noise_variance
            )

        # Step 2: Contrast enhancement
        if self.enhance_contrast:
            result = self._apply_clahe(result)

        # Step 3: Resize if target size specified
        if self.target_size is not None:
            result = cv2.resize(result, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Step 4: Normalize if requested
        if self.normalize:
            result = result.astype(np.float32) / 255.0

        return result

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE enhances local contrast, making ships more distinguishable from
        the water background after speckle filtering.
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self._clahe.apply(image)

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Process a SAR image file.

        Args:
            input_path: Path to input image
            output_path: Path to save processed image (optional)

        Returns:
            Processed image array
        """
        input_path = Path(input_path)
        image = cv2.imread(str(input_path))

        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")

        processed = self.process(image)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)

        return processed

    def process_batch(
        self,
        image_paths: list,
        output_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> list:
        """
        Process multiple SAR images.

        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save processed images
            show_progress: Print progress updates

        Returns:
            List of processed image arrays
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            if show_progress and (i + 1) % 10 == 0:
                print(f"[INFO] Processing: {i + 1}/{total}")

            if output_dir is not None:
                out_path = Path(output_dir) / Path(path).name
            else:
                out_path = None

            processed = self.process_file(path, out_path)
            results.append(processed)

        return results

    def compare_before_after(
        self,
        image: np.ndarray,
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Create a side-by-side comparison of original and processed image.

        Useful for visualizing the effect of speckle filtering.

        Args:
            image: Input SAR image
            save_path: Path to save comparison image

        Returns:
            Combined before/after image
        """
        processed = self.process(image)

        # Ensure same dimensions for concatenation
        if len(image.shape) == 2:
            original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if len(processed.shape) == 2:
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed
        else:
            original = image
            processed_bgr = processed

        # Add labels
        h, w = original.shape[:2]
        labeled_orig = original.copy()
        labeled_proc = processed_bgr.copy()

        cv2.putText(labeled_orig, "Original (Speckled)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(labeled_proc, "Lee Filtered", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Concatenate horizontally
        comparison = np.hstack([labeled_orig, labeled_proc])

        if save_path is not None:
            cv2.imwrite(str(save_path), comparison)

        return comparison


def main():
    """Demo the preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAR Image Speckle Filtering Demo"
    )
    parser.add_argument(
        'input', type=str,
        help='Path to input SAR image'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path for output image'
    )
    parser.add_argument(
        '--window-size', type=int, default=7,
        help='Lee filter window size'
    )
    parser.add_argument(
        '--enhanced', action='store_true',
        help='Use enhanced Lee filter'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Create before/after comparison'
    )

    args = parser.parse_args()

    preprocessor = SARPreprocessor(
        lee_window_size=args.window_size,
        use_enhanced_lee=args.enhanced,
        enhance_contrast=True
    )

    image = cv2.imread(args.input)
    if image is None:
        print(f"[ERROR] Could not load image: {args.input}")
        return

    if args.compare:
        result = preprocessor.compare_before_after(image, args.output)
        print(f"[INFO] Comparison saved to: {args.output or 'comparison.jpg'}")
    else:
        result = preprocessor.process(image)
        output_path = args.output or f"filtered_{Path(args.input).name}"
        cv2.imwrite(output_path, result)
        print(f"[INFO] Filtered image saved to: {output_path}")


if __name__ == "__main__":
    main()
