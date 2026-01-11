"""
Converts on-line spectal image plots to digital data to be used for reference 
data-sets.  Code converts image to digital, formats in comma seperated standard format
(wavelength, intensity) and plots new data (picture vs. digitized version) for comparison.

Data is formatter in ascending range of wavelength

This version is hard-coded for 390-1200nm.  Use graph_to...._variable.py for 
user specified range.

Works remarkably well, a slight loss of resolution due to limiting to 100 datapoints
across spectrum but no diagnostic loss at all.

Version to fix peak clipping

MODIFIED VERSION: Batch processes all image files in the input folder

Be sure to set the paths for OG image plot and new saved output throughtout code where paths
appear.  ALL PATHS MUST BE CHANGED IN THE CODE FOR IT TO WORK!

=====================

"""

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt
import datetime
import traceback

# ===== CONFIGURATION =====
# Set your input folder containing mineral graph images here
INPUT_FOLDER = "/path/of/original/graph" # <- set path to snagged image locally
# Set your output folder for saving extracted data and visualizations
OUTPUT_FOLDER = "/path/of/output/digital/graph"

# ========================

def extract_spectrum_from_image(image_path, debug=False):
    """
    Extract spectral data from an image of a mineral reflection spectrum.
    
    Args:
        image_path: Path to the image file
        debug: If True, save debug images and plots
        
    Returns:
        wavelengths: Array of 100 wavelength values from 400nm to 1200nm
        intensities: Array of 100 intensity values from 0 to 1
        img: Original image
    """
    # Create debug folder if debug is enabled
    if debug:
        debug_folder = os.path.join(OUTPUT_FOLDER, "/path/of/output/digital/graph")
        os.makedirs(debug_folder, exist_ok=True)
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image was loaded properly
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Also extract blue channel (BGR format, so blue is at index 0)
    blue_channel = img[:, :, 0].copy()
    
    # --- OPTIMIZED APPROACH: Focus primarily on blue signal without over-filtering ---
    
    # Enhance blue relative to other channels - very permissive
    enhanced_blue = np.zeros_like(img[:,:,0])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            b, g, r = img[y, x]
            # Look for areas where blue is stronger than other channels
            # Using an even smaller difference threshold to be more inclusive
            if b > max(r, g) + 3:  # Very minimal threshold
                enhanced_blue[y, x] = 255
    
    # If enhanced_blue doesn't have enough points, fall back to inverse grayscale
    if np.sum(enhanced_blue > 0) < width * 0.15:  # Even lower threshold
        print("Using grayscale detection as fallback")
        # Invert grayscale (line will be bright on dark background)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        # Use enhanced blue
        _, binary = cv2.threshold(enhanced_blue, 20, 255, cv2.THRESH_BINARY)  # Lower threshold
    
    # Apply minimal morphological operations to clean up the binary image
    # Using a smaller kernel to preserve even more fine details
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "/path/of/original/graph/debug_binary.png"), binary)
        cv2.imwrite(os.path.join(debug_folder, "/path/of/original/graph/debug_enhanced_blue.png"), enhanced_blue)
    
    # --- IMPROVED: Text handling via ROI detection ---
    # Crop out the potential legend area (top portion of the image)
    legend_region_height = int(height * 0.1)  # Top 10% of image height
    main_graph_binary = binary.copy()
    main_graph_binary[:legend_region_height, :] = 0  # Zero out the top portion
    
    if debug:
        cv2.imwrite(os.path.join(debug_folder, "/path/of/original/graph/debug_main_graph_binary.png"), main_graph_binary)
    
    # Find the curve by scanning each column in the main graph area
    points = []
    last_valid_y = None
    # Even looser jump threshold - only filter extreme outliers
    max_jump_allowed = height * 0.12  # Increased from 8% to 12%
    
    for x in range(width):
        # Find all y-coordinates for this x where binary is non-zero
        y_values = np.where(main_graph_binary[:, x] > 0)[0]
        if len(y_values) > 0:
            # Use the minimum y-value (closest to top of image)
            # This will favor the actual spectrum line over other elements
            y = np.min(y_values)
            
            # Filter out only the most extreme jumps
            if last_valid_y is not None:
                if abs(y - last_valid_y) > max_jump_allowed:
                    print(f"Filtering outlier at x={x}, y={y} (last_y={last_valid_y})")
                    continue
            
            points.append((x, y))
            last_valid_y = y
    
    # Convert to numpy array for easier manipulation
    if not points:
        # More detailed error to help diagnose issues
        raise ValueError(
            "Could not extract any points from the spectrum line. "
            "Try adjusting the thresholds or check if the image format is correct."
        )
    
    # Convert to numpy array
    points = np.array(points)
    
    # If we have very few points, try different detection approach
    if len(points) < width * 0.15:  # Even lower threshold
        print(f"Warning: Detected only {len(points)} points, trying alternative approach...")
        
        # Alternative approach: adaptive threshold on grayscale image
        binary_alt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        if debug:
            cv2.imwrite(os.path.join(debug_folder, "/path/of/original/graph/debug_binary_alt.png"), binary_alt)
        
        # Try to extract points again
        alt_points = []
        for x in range(width):
            y_values = np.where(binary_alt[:, x] > 0)[0]
            if len(y_values) > 0:
                alt_points.append((x, np.min(y_values)))
        
        # If alternative approach found more points, use those instead
        if len(alt_points) > len(points):
            print(f"Alternative approach found {len(alt_points)} points, using those instead.")
            points = np.array(alt_points)
    
    # Sort by x-coordinate to ensure order
    points = points[points[:, 0].argsort()]
    
    # Apply a very minimal smoothing to reduce noise while preserving features
    # We're using a much smaller window to preserve even more fine details
    try:
        window_length = min(5, len(points) - 2)  # Reduced from 9 to 5
        # Make sure window_length is odd
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:  # Check if we have enough points for smoothing
            points[:, 1] = savgol_filter(points[:, 1], window_length, 2)
    except Exception as e:
        print(f"Warning: Smoothing failed, using raw points: {e}")
    
    # Debug: plot extracted points on original image
    if debug:
        debug_img = img.copy()
        for x, y in points:
            cv2.circle(debug_img, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(debug_folder, "/path/of/original/graph/debug_extraction.png"), debug_img)
        
        # Also plot extracted curve
        plt.figure(figsize=(10, 6))
        plt.plot(points[:, 0], points[:, 1])
        plt.title("Extracted Curve (Image Coordinates)")
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.savefig(os.path.join(debug_folder, "/path/of/original/graph/debug_curve.png"))
        plt.close()
    
    # ----- MINIMAL EDGE TRIMMING -----
    # Remove only the extreme edges to avoid cutting spectral features
    trim_percent = 0.015  # Reduced from 0.02 to 0.015 (1.5%)
    
    # Calculate the number of points to trim from each side
    num_points = len(points)
    trim_count = int(num_points * trim_percent)
    
    # Only trim if we have enough points to do so
    if num_points > 30:  # Ensure we don't trim too much from small datasets
        if trim_count > 0:
            points = points[trim_count:-trim_count]
            print(f"Trimmed {trim_count} points from each side to remove potential axes")
    
    # Get the range of x values after trimming
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    
    # Calculate wavelength based on position within the graph area
    # IMPORTANT: Graph runs from 1200nm (left) to 400nm (right)
    wavelength_min, wavelength_max = 400, 1200  # Using 400nm as minimum wavelength
    
    # Create more precise wavelength mapping using piece-wise approach
    # This accounts for potential non-linear scaling in the source graph
    
    # Three-segment correction: different correction factors for different regions
    # This helps better align features across the spectrum
    x_values = []
    for x in points[:, 0]:
        normalized_pos = (x - min_x) / (max_x - min_x)
        
        # Apply region-specific correction factors - fine-tuned values
        if normalized_pos < 0.33:  # Left region (1200-930nm)
            corrected_pos = normalized_pos * 0.95  # Refined correction factor
        elif normalized_pos < 0.66:  # Middle region (930-660nm)
            corrected_pos = 0.33 * 0.95 + (normalized_pos - 0.33) * 1.0  # Normal in middle
        else:  # Right region (660-400nm)
            corrected_pos = 0.33 * 0.95 + 0.33 * 1.0 + (normalized_pos - 0.66) * 1.05  # Refined correction
        
        # Map to wavelength (reversed: high on left, low on right)
        wavelength = wavelength_max - corrected_pos * (wavelength_max - wavelength_min)
        x_values.append(wavelength)
    
    # Convert to numpy array
    x_values = np.array(x_values)

    # ----- IMPROVED INTENSITY NORMALIZATION TO PREVENT PEAK CLIPPING -----
    # For y, we need to invert since image coordinates have origin at top-left
    y_values = 1 - (points[:, 1] / height)

    # Store original values before any processing
    original_y = y_values.copy()

    # First pass: Basic min-max scaling to get a rough idea of the distribution
    if np.max(y_values) > np.min(y_values):
        prelim_scaled = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    else:
        prelim_scaled = y_values

    # Detect plateaus by looking at the density of points near the top of the range
    sorted_vals = np.sort(prelim_scaled)
    top_20_percent = sorted_vals[int(0.8 * len(sorted_vals)):]
    max_val = np.max(top_20_percent)
    near_max_count = np.sum(top_20_percent > max_val * 0.95)

    # Log information about potential plateaus
    print(f"Top range analysis: {near_max_count} points near maximum")
    print(f"Max value: {max_val:.4f}, Min value: {np.min(prelim_scaled):.4f}")

    # Determine if this is a plateau-type spectrum
    has_plateau = near_max_count > len(y_values) * 0.1  # If more than 10% are near max

    # MODIFIED: Use different normalization approaches based on spectrum type
    if has_plateau:
        print("Detected plateau-type spectrum, applying special normalization")
        
        # For plateau spectra, we want to compress the upper range a bit
        # to preserve details rather than flattening everything to 1.0
        
        # Find 95th percentile value to use as scaling reference
        p95_value = np.percentile(y_values, 95)
        
        # Scale so that the 95th percentile maps to 0.95
        # This leaves room for peaks to go higher without clipping
        scaling_factor = 0.90 / p95_value  # Changed from 0.95 to 0.90 to allow more headroom
        y_values = y_values * scaling_factor
        
        # Then shift to ensure minimum is at 0
        min_val = np.min(y_values)
        y_values = y_values - min_val
        
        # Finally normalize to 0-1 range
        max_val = np.max(y_values)
        if max_val > 0:
            # Use slightly lower factor to ensure peaks don't get clipped at exactly 1.0
            y_values = y_values / max_val * 0.995  # Scale to 0.995 instead of 1.0
    else:
        print("Standard spectrum with distinct peaks, applying peak-preserving normalization")
        # NEW: For spectra with distinct peaks, use a different approach that preserves peaks
        
        # First normalize to rough 0-1 range
        if np.max(y_values) > np.min(y_values):
            y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
        
        # NEW: Look for potential peak clipping by checking density at the top
        top_values = y_values[y_values > 0.95]
        potential_clipping = len(top_values) > len(y_values) * 0.05  # More than 5% at the very top
        
        if potential_clipping:
            print("Potential peak clipping detected, applying peak preservation scaling")
            # Compress the entire range slightly to ensure peaks aren't clipped
            y_values = y_values * 0.95
            
            # Then re-normalize to ensure we utilize the full 0-0.995 range
            if np.max(y_values) > 0:
                y_values = (y_values / np.max(y_values)) * 0.995

    # For both approaches - ensure no value exceeds 0.995
    # This helps avoid exactly 1.0 values while still preserving peak shapes
    if np.max(y_values) > 0.995:
        print("Applying final adjustment to avoid 1.0 values")
        y_values = y_values * (0.995 / np.max(y_values))

    # Debug: Print range after normalization
    print(f"Final intensity range: {np.min(y_values):.4f} - {np.max(y_values):.4f}")
    
    # end modified section

    # Debug: Save comparison of original and normalized values
    if debug:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(original_y)), original_y, label='Original', alpha=0.5)
        plt.scatter(range(len(y_values)), y_values, label='Normalized', alpha=0.5)
        plt.title("/path/of/original/graph/Intensity Normalization Comparison")
        plt.xlabel("Point Index")
        plt.ylabel("Intensity Value")
        plt.legend()
        plt.savefig(os.path.join(debug_folder, "/path/of/original/graph/normalization_comparison.png"))
        plt.close()
    
    # Sort by wavelength (ascending order from 400nm to 1200nm)
    sort_indices = np.argsort(x_values)
    x_values = x_values[sort_indices]
    y_values = y_values[sort_indices]
    
    print(f"Data range detected: {min(x_values):.2f}nm to {max(x_values):.2f}nm")
    
    # Resample to get exactly 100 points
    try:
        # Use standard range of 400-1200nm for output
        x_new = np.linspace(400, 1200, 100)
        
        # Try cubic interpolation first (best for smooth curves)
        f = interp1d(x_values, y_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
        y_new = f(x_new)
        print("Successfully used cubic interpolation")
    except Exception as e:
        print(f"Warning: Cubic interpolation failed: {e}")
        try:
            # Fall back to linear interpolation
            print("Falling back to linear interpolation")
            f = interp1d(x_values, y_values, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_new = f(x_new)
            print("Successfully used linear interpolation")
        except Exception as e:
            print(f"Warning: Linear interpolation also failed: {e}")
            # Last resort: nearest neighbor interpolation
            print("Falling back to nearest neighbor interpolation")
            f = interp1d(x_values, y_values, kind='nearest', bounds_error=False, fill_value="extrapolate")
            y_new = f(x_new)
            print("Successfully used nearest neighbor interpolation")

    # MODFICATION START:
    
    # MODIFIED: Better handle the final normalization to prevent peak clipping
    # Check if values are outside the expected range
    if np.max(y_new) > 1.0 or np.min(y_new) < 0.0:
        print("Re-normalizing interpolated values")
        
        # Handle negative values by shifting everything up
        if np.min(y_new) < 0.0:
            y_new = y_new - np.min(y_new)
        
        # Scale peaks carefully to preserve their relative heights
        if np.max(y_new) > 1.0:
            # Preserve relative peak heights by scaling all values proportionally
            scaling_factor = 0.995 / np.max(y_new)  # Use 0.995 to avoid exact 1.0 values
            y_new = y_new * scaling_factor
            print(f"Applied scaling factor of {scaling_factor:.4f} to preserve peak heights")
    else:
        # Even if values are in range, check for potential peak clipping
        potential_clipping = np.sum(y_new > 0.99) > 5  # More than 5 points very close to 1.0
        
        if potential_clipping:
            print("Adjusting interpolated values to avoid peak clipping")
            # First find points that might be clipped
            high_points = y_new > 0.99
            # Scale everything slightly to make room for peaks
            scaling_factor = 0.995 / np.max(y_new)
            y_new = y_new * scaling_factor
    
    print(f"Final intensity range after resampling: {np.min(y_new):.4f} - {np.max(y_new):.4f}")
    
    # Debug: plot resampled data
    if debug:
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'b.', label='Extracted')
        plt.plot(x_new, y_new, 'r-', label='Resampled')
        plt.title("Resampled Curve (Actual Values)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.savefig(os.path.join(debug_folder, "debug_resampled.png"))
        plt.close()
    
    return x_new, y_new, img  # Return original image for visualization

def save_spectrum_data(wavelengths, intensities, mineral_name, output_folder=OUTPUT_FOLDER):
    """
    Save the extracted spectrum data to CSV and TXT files.
    
    Args:
        wavelengths: Array of wavelength values
        intensities: Array of intensity values
        mineral_name: Name of the mineral
        output_folder: Folder to save the output file
        
    Returns:
        output_files: Dictionary with paths to the saved files
        safe_name: Sanitized version of the mineral name for use in filenames
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create filename based on mineral name
    safe_name = "".join(c if c.isalnum() else "_" for c in mineral_name)
    
    """
    # uncomment to enable .txt format save
    # Save as TXT file
    txt_file = os.path.join(output_folder, f"{safe_name}_spectrum.txt")
    with open(txt_file, 'w') as f:
        # Write mineral name as first line
        f.write(f"{mineral_name}\n")
        
        # Write wavelength and intensity pairs
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f},{intensity:.6f}\n")
    """
    # Save as CSV file
    csv_file = os.path.join(output_folder, f"{safe_name}_spectrum.csv")
    with open(csv_file, 'w') as f:
        # Write CSV header
        f.write("wavelength,intensity\n")
        
        # Write wavelength and intensity pairs
        for wl, intensity in zip(wavelengths, intensities):
            f.write(f"{wl:.2f},{intensity:.6f}\n")
    
    print(f"Data saved to:")
    # print(f"  - TXT: {txt_file}")
    print(f"  - CSV: {csv_file}")
    
    return {"csv": csv_file}, safe_name # use .txt save option below if enabled
    # return {"txt": txt_file, "csv": csv_file}, safe_name

def visualize_extraction(wavelengths, intensities, original_img, mineral_name, output_folder=OUTPUT_FOLDER):
    """
    Create a visual comparison of original and extracted data.
    
    Args:
        wavelengths: Array of wavelength values
        intensities: Array of intensity values
        original_img: Original image
        mineral_name: Name of the mineral
        output_folder: Folder to save the output file
        
    Returns:
        output_path: Path to the saved visualization
    """
    # Create a safe filename
    safe_name = "".join(c if c.isalnum() else "_" for c in mineral_name)
    
    # Create the figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original image
    axs[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    
    # Display with correct wavelength orientation for original image
    # Add x-axis labels to show decreasing wavelength (1200nm to 400nm) from left to right
    width = original_img.shape[1]
    tick_positions = np.linspace(0, width, 5)
    tick_labels = np.linspace(1200, 400, 5).astype(int)
    axs[0].set_xticks(tick_positions)
    axs[0].set_xticklabels(tick_labels)
    axs[0].set_xlabel("Wavelength (nm) - Original Scale (Descending)")
    axs[0].set_ylabel("Image Y-coordinate")
    
    # Plot extracted data
    axs[1].plot(wavelengths, intensities, 'b-', linewidth=2)
    axs[1].set_title(f"Extracted Spectrum for {mineral_name}")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Intensity")
    axs[1].grid(True)
    
    # Set x-axis limits to standard range
    axs[1].set_xlim(400, 1200)

    # MODIFIED: Set y-axis limits with added padding to ensure peaks are visible
    # Add more padding at the top (10%) to ensure no peaks are cut off in visualization
    max_intensity = max(intensities)
    axs[1].set_ylim(-0.05, max(1.05, max_intensity * 1.1))
    
    # Add text with information about the extraction
    min_wl, max_wl = min(wavelengths), max(wavelengths)
    min_int, max_int = min(intensities), max(intensities)
    info_text = (
        f"Extracted {len(wavelengths)} points\n"
        f"Wavelength range: {min_wl:.1f} - {max_wl:.1f} nm\n"
        f"Intensity range: {min_int:.3f} - {max_int:.3f}"
    )
    axs[1].text(
        0.05, 0.05, info_text,
        transform=axs[1].transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{safe_name}_comparison.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def list_image_files(folder_path):
    """
    List all image files in the given folder.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        list: List of image file paths
    """
    # Common image extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # Get all files with image extensions
    image_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    return image_files

def batch_process_folder(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, debug=False):
    """
    Process all image files in the input folder and convert them to spectral data.
    
    Args:
        input_folder: Path to the folder containing images
        output_folder: Folder to save the output files
        debug: If True, save debug images and plots
    
    Returns:
        successful_files: List of successfully processed files
        failed_files: List of files that failed to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a log file for the batch process
    log_file_path = os.path.join(output_folder, "batch_process_log.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write("=== Batch Spectrum Conversion Log ===\n")
        log_file.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Input folder: {input_folder}\n")
        log_file.write(f"Output folder: {output_folder}\n\n")
    
    # Get all image files in the input folder
    image_files = list_image_files(input_folder)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        with open(log_file_path, 'a') as log_file:
            log_file.write("No image files found in the input folder.\n")
        return [], []
    
    print(f"Found {len(image_files)} image files to process.")
    
    # Lists to track processing results
    successful_files = []
    failed_files = []
    
    # Process each image file
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        mineral_name = os.path.splitext(filename)[0]  # Use filename without extension as mineral name
        
        print(f"\n[{i+1}/{len(image_files)}] Processing {filename}...")
        
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"\n[{i+1}/{len(image_files)}] Processing {filename}\n")
            log_file.write(f"  Mineral name: {mineral_name}\n")
        
        try:
            # Extract spectrum data from image
            wavelengths, intensities, original_img = extract_spectrum_from_image(image_path, debug=debug)
            
            # Save data to file
            output_files, safe_name = save_spectrum_data(wavelengths, intensities, mineral_name)
            
            # Create visualization for quality assessment
            viz_path = visualize_extraction(wavelengths, intensities, original_img, mineral_name)
            
            print(f"Successfully extracted and saved spectrum for {mineral_name}")
            
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"  SUCCESS: Data saved to {output_files['csv']}\n")
                log_file.write(f"  Visualization saved to {viz_path}\n")
            
            successful_files.append(filename)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Get full traceback
            tb = traceback.format_exc()
            
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"  ERROR: Failed to process {filename}\n")
                log_file.write(f"  Error details: {str(e)}\n")
                log_file.write(f"  Traceback: {tb}\n")
            
            failed_files.append(filename)
    
    # Write summary to log file
    with open(log_file_path, 'a') as log_file:
        log_file.write("\n=== Summary ===\n")
        log_file.write(f"Total files: {len(image_files)}\n")
        log_file.write(f"Successfully processed: {len(successful_files)}\n")
        log_file.write(f"Failed: {len(failed_files)}\n")
        
        if failed_files:
            log_file.write("\nFailed files:\n")
            for file in failed_files:
                log_file.write(f"  - {file}\n")
        
        log_file.write(f"\nFinished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Print summary
    print("\n=== Batch Processing Summary ===")
    print(f"Total files: {len(image_files)}")
    print(f"Successfully processed: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    print(f"\nLog file saved to: {log_file_path}")
    
    return successful_files, failed_files

def main():
    print("===== Mineral Reflection Spectra Batch Converter =====")
    print("This program extracts spectral data from all images in the input folder.")
    print("Each spectrum is converted to 100 data points in the range 400-1200nm.")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Process all images in the input folder
    print("\nStarting batch processing...")
    successful_files, failed_files = batch_process_folder(INPUT_FOLDER, OUTPUT_FOLDER, debug=True)
    
    print("\nBatch processing complete!")
    
    # If all files were processed successfully
    if not failed_files:
        print("All files were processed successfully!")
    else:
        print(f"{len(failed_files)} files failed to process. Check the log file for details.")
    
    print(f"\nProcessed data and visualizations are saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()

