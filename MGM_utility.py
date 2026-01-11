"""
MGM (modified gausian modeling) is a technique to parse spectral data and analyze
in ways that allow for more discrete identification of various spectral signatures
(see: https://bpb-us-w2.wpmucdn.com/sites.brown.edu/dist/b/323/files/2022/02/LPSC99MGM.pdf)

Using MGM, in theory, because MGM allows for the discrete isolation of specific 
chemical (or color) bands in the spectra, one does not need to do complex algrythic matching
if the sample has sufficient signal and if the reference signature is known.

This code demonstrates MGM analsys on raw spectral files, converting .csv to 
required .asc format (simple 2-column: wavelength, reflectance)
and generate a startup file with initial guesses for:

Number of absorption bands expected
Initial band centers (wavelengths)
Initial band widths
Initial band strengths
Continuum parameters (slope, offset)

Code written for roughly 350-1000nm but should dynamically adjust to the range
and ignore headers.



"""


# ============================================================================
# QUICK START COMMANDS
# ============================================================================

# To analyze your real data (make sure SPECTRUM_PATH is set correctly above):
# result = analyze_real_spectrum()

# To run the synthetic example:
# example_result = example_usage()

# To analyze a different file without changing the path variable:
# result = analyze_real_spectrum("path/to/different/file.csv")

# ============================================================================
# AUTO-RUN SECTION - Uncomment what you want to run automatically
# ============================================================================

# ============================================================================
# CONFIGURATION - Change these paths/settings as needed
# ============================================================================
SPECTRUM_PATH = "/Users/paul/sensor_output_files/calibrated_samples_smoothed/corundumWhiteGrayCatsC2_calibrated_normalized_interpolated_100pts_smoothed.csv"  # Change this to your CSV file path
N_BANDS = 3                          # Number of absorption bands to fit
FIGURE_SIZE = (12, 8)               # Plot size (width, height)

# Mac-specific matplotlib backend setup
import matplotlib
matplotlib.use('TkAgg')  # Forces plots to open in separate windows on Mac
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Qt5Agg')

class ModifiedGaussianModel:
    """
    Python implementation of Modified Gaussian Model (MGM) for mineral spectroscopy
    Based on Sunshine et al. (1990) methodology
    """
    
    def __init__(self):
        self.wavelength = None
        self.reflectance = None
        self.energy = None  # 1/wavelength for MGM calculations
        self.log_reflectance = None
        self.fitted_params = None
        self.fitted_spectrum = None
        self.individual_bands = None
        self.continuum = None
        self.rms_error = None
        
    def load_spectrum(self, csv_file=None, wavelength=None, reflectance=None):
        """
        Load spectral data from CSV file or arrays
        
        Parameters:
        csv_file: path to CSV file with columns [wavelength, reflectance]
        wavelength: array of wavelengths (nm)
        reflectance: array of reflectance values
        """
        if csv_file is not None:
            data = pd.read_csv(csv_file)
            self.wavelength = data.iloc[:, 0].values
            self.reflectance = data.iloc[:, 1].values
        else:
            self.wavelength = np.array(wavelength)
            self.reflectance = np.array(reflectance)
        
        # Convert to energy units (1/wavelength) - handle different wavelength ranges
        self.energy = 10000.0 / self.wavelength  # Convert nm to wavenumber (cm^-1)
        self.log_reflectance = np.log(np.clip(self.reflectance, 1e-6, 1.0))  # Avoid log(0)
        
        print(f"Loaded spectrum: {len(self.wavelength)} points")
        print(f"Wavelength range: {self.wavelength.min():.1f} - {self.wavelength.max():.1f} nm")
        print(f"Energy range: {self.energy.min():.1f} - {self.energy.max():.1f} cm^-1")
        
        # Check for common wavelength ranges and give guidance
        if self.wavelength.max() < 1200:
            print("Note: This appears to be a UV-VIS spectrum (350-1000nm range)")
            print("Typical absorption bands in this range: Fe3+ charge transfer, crystal field transitions")
        elif self.wavelength.max() > 2000:
            print("Note: This appears to be a UV-VIS-NIR spectrum with full range")
            print("Typical absorption bands: Fe2+ electronic transitions, overtones, combinations")
        
    def modified_gaussian(self, energy, center, width, strength):
        """
        Modified Gaussian function in energy space
        
        Parameters:
        energy: energy values (1/wavelength)
        center: band center in energy units
        width: band width parameter
        strength: band strength (negative for absorption)
        """
        # Modified Gaussian with asymmetric tails
        gamma = 0.5 * width
        delta = (energy - center) / gamma
        
        # Use error function for the asymmetric shape
        result = strength * np.exp(-delta**2) * (1 + erf(delta))
        return result
    
    def continuum_function(self, energy, offset, slope):
        """
        Linear continuum in energy space
        """
        return offset + slope * energy
    
    def full_model(self, params, energy):
        """
        Full MGM model: continuum + sum of modified Gaussians
        
        Parameters:
        params: array of parameters [offset, slope, center1, width1, strength1, ...]
        energy: energy values
        """
        # Extract continuum parameters
        offset = params[0]
        slope = params[1]
        
        # Start with continuum
        model = self.continuum_function(energy, offset, slope)
        
        # Add each Gaussian band
        n_bands = (len(params) - 2) // 3
        for i in range(n_bands):
            idx = 2 + i * 3
            center = params[idx]
            width = params[idx + 1]
            strength = params[idx + 2]
            model += self.modified_gaussian(energy, center, width, strength)
        
        return model
    
    def residuals(self, params, energy, data):
        """Calculate residuals for least squares fitting"""
        model = self.full_model(params, energy)
        return model - data
    
    def auto_initialize_parameters(self, n_bands=3):
        """
        Automatically initialize parameters based on spectrum features
        
        Parameters:
        n_bands: number of absorption bands to fit
        """
        # Initialize continuum with simple linear fit to spectrum endpoints
        continuum_offset = np.mean([self.log_reflectance[0], self.log_reflectance[-1]])
        continuum_slope = (self.log_reflectance[-1] - self.log_reflectance[0]) / (self.energy[-1] - self.energy[0])
        
        # Ensure continuum parameters are within reasonable bounds
        continuum_offset = np.clip(continuum_offset, -2, 2)
        continuum_slope = np.clip(continuum_slope, -0.005, 0.005)
        
        # Find approximate absorption band positions
        # Remove rough continuum and find minima
        rough_continuum = continuum_offset + continuum_slope * self.energy
        residual = self.log_reflectance - rough_continuum
        
        # Find n_bands deepest minima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-residual, prominence=0.01, distance=len(residual)//10)
        
        if len(peaks) < n_bands:
            # If not enough peaks found, distribute evenly across spectrum
            band_energies = np.linspace(self.energy.min() * 1.1, self.energy.max() * 0.9, n_bands)
        else:
            # Use the deepest peaks
            peak_depths = -residual[peaks]
            sorted_indices = np.argsort(peak_depths)[::-1]
            selected_peaks = peaks[sorted_indices[:n_bands]]
            band_energies = self.energy[selected_peaks]
        
        # Initialize parameters
        initial_params = [continuum_offset, continuum_slope]
        
        for energy_center in band_energies:
            # Ensure energy centers are within bounds
            energy_center = np.clip(energy_center, self.energy.min(), self.energy.max())
            initial_params.extend([
                energy_center,           # center
                0.3,                    # width (reasonable default)
                -0.1                    # strength (negative for absorption)
            ])
        
        return np.array(initial_params)
    
    def fit_spectrum(self, n_bands=3, initial_params=None, max_iterations=100):
        """
        Fit MGM to the loaded spectrum
        
        Parameters:
        n_bands: number of absorption bands to fit
        initial_params: manual initial parameter guess (optional)
        max_iterations: maximum fitting iterations
        """
        if self.energy is None:
            raise ValueError("No spectrum loaded. Use load_spectrum() first.")
        
        # Initialize parameters
        if initial_params is None:
            initial_params = self.auto_initialize_parameters(n_bands)
        
        print(f"Fitting {n_bands} bands with MGM...")
        # Print initial parameters for debugging
        print(f"Initial parameters: continuum offset={initial_params[0]:.3f}, slope={initial_params[1]:.6f}")
        for i in range(n_bands):
            idx = 2 + i * 3
            print(f"Band {i+1}: center={10000/initial_params[idx]:.1f}nm, width={initial_params[idx+1]:.3f}, strength={initial_params[idx+2]:.3f}")
        
        # Set parameter bounds (reasonable physical constraints)
        lower_bounds = []
        upper_bounds = []
        
        # Continuum bounds
        lower_bounds.extend([-3, -0.008])  # offset, slope
        upper_bounds.extend([3, 0.008])
        
        # Band bounds
        for i in range(n_bands):
            lower_bounds.extend([
                self.energy.min() * 0.95,  # center: within energy range
                0.05,                      # width: must be positive
                -1.5                       # strength: negative for absorption
            ])
            upper_bounds.extend([
                self.energy.max() * 1.05,  # center: within energy range
                1.5,                       # width: reasonable maximum
                0.3                        # strength: allow weak positive features
            ])
        
        # Perform least squares fitting
        try:
            result = least_squares(
                self.residuals,
                initial_params,
                args=(self.energy, self.log_reflectance),
                bounds=(lower_bounds, upper_bounds),
                max_nfev=max_iterations * len(initial_params)
            )
            
            self.fitted_params = result.x
            self.fitted_spectrum = self.full_model(self.fitted_params, self.energy)
            
            # Calculate RMS error
            residuals = self.fitted_spectrum - self.log_reflectance
            self.rms_error = np.sqrt(np.mean(residuals**2))
            
            # Extract individual components
            self._extract_components()
            
            print(f"Fitting completed successfully!")
            print(f"RMS Error: {self.rms_error:.4f}")
            print(f"Iterations: {result.nfev}")
            
            return True
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            return False
    
    def _extract_components(self):
        """Extract individual bands and continuum from fitted parameters"""
        if self.fitted_params is None:
            return
        
        # Extract continuum
        offset = self.fitted_params[0]
        slope = self.fitted_params[1]
        self.continuum = self.continuum_function(self.energy, offset, slope)
        
        # Extract individual bands
        n_bands = (len(self.fitted_params) - 2) // 3
        self.individual_bands = []
        
        for i in range(n_bands):
            idx = 2 + i * 3
            center = self.fitted_params[idx]
            width = self.fitted_params[idx + 1]
            strength = self.fitted_params[idx + 2]
            
            band = self.modified_gaussian(self.energy, center, width, strength)
            self.individual_bands.append(band)
    
    def get_band_parameters(self):
        """
        Return fitted band parameters in a readable format
        """
        if self.fitted_params is None:
            return None
        
        # Convert back to wavelength for reporting
        results = {
            'continuum': {
                'offset': self.fitted_params[0],
                'slope': self.fitted_params[1]
            },
            'bands': []
        }
        
        n_bands = (len(self.fitted_params) - 2) // 3
        for i in range(n_bands):
            idx = 2 + i * 3
            center_energy = self.fitted_params[idx]
            width = self.fitted_params[idx + 1]
            strength = self.fitted_params[idx + 2]
            
            # Convert back to wavelength from wavenumber
            center_wavelength = 10000.0 / center_energy  # Convert cm^-1 back to nm
            
            # Calculate band area (integrated absorption)
            band_area = abs(strength * width * np.sqrt(np.pi))
            
            results['bands'].append({
                'band_number': i + 1,
                'center_wavelength_nm': center_wavelength,
                'center_energy': center_energy,
                'width': width,
                'strength': strength,
                'area': band_area
            })
        
        return results
    
    def plot_results(self, figsize=(12, 8)):
        """
        Plot the original spectrum, fitted model, and individual components
        """
        if self.fitted_spectrum is None:
            print("No fitted model to plot. Run fit_spectrum() first.")
            return
        
        # Clear any existing plots
        plt.close('all')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Top plot: Original vs fitted in reflectance space
        ax1.plot(self.wavelength, self.reflectance, 'k-', label='Original Data', linewidth=2)
        ax1.plot(self.wavelength, np.exp(self.fitted_spectrum), 'r-', 
                label=f'MGM Fit (RMS: {self.rms_error:.4f})', linewidth=2)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Reflectance')
        ax1.set_title('MGM Spectral Deconvolution Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Components in log reflectance space
        ax2.plot(self.wavelength, self.log_reflectance, 'k-', label='Original (log)', linewidth=2)
        ax2.plot(self.wavelength, self.fitted_spectrum, 'r-', label='Total Fit', linewidth=2)
        ax2.plot(self.wavelength, self.continuum, 'b--', label='Continuum', linewidth=1.5)
        
        # Plot individual bands
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for i, band in enumerate(self.individual_bands):
            color = colors[i % len(colors)]
            ax2.plot(self.wavelength, self.continuum + band, '--', 
                    color=color, label=f'Band {i+1}', linewidth=1.5)
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Log Reflectance')
        ax2.set_title('MGM Components (Log Reflectance Space)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Force display with multiple methods for Mac compatibility
        plt.draw()
        plt.pause(0.001)  # Small pause to ensure rendering
        plt.show(block=False)  # Non-blocking show
        
        # Also try to bring window to front
        try:
            fig.canvas.manager.window.wm_attributes('-topmost', 1)
            fig.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass
        
        # Print band parameters
        params = self.get_band_parameters()
        if params:
            print("\nFitted Band Parameters:")
            print("=" * 60)
            print(f"Continuum - Offset: {params['continuum']['offset']:.4f}, "
                  f"Slope: {params['continuum']['slope']:.6f}")
            print("=" * 60)
            for band in params['bands']:
                print(f"Band {band['band_number']}: "
                      f"Center = {band['center_wavelength_nm']:.1f} nm, "
                      f"Width = {band['width']:.3f}, "
                      f"Strength = {band['strength']:.3f}, "
                      f"Area = {band['area']:.3f}")
        
        # Keep window open
        print("\nPlot window should be visible. Press Enter to continue or Ctrl+C to exit.")
        try:
            input()
        except KeyboardInterrupt:
            pass

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_real_spectrum(file_path=None):
    """
    Analyze a real spectrum from CSV file
    """
    if file_path is None:
        file_path = SPECTRUM_PATH
    
    try:
        print(f"Loading spectrum from: {file_path}")
        mgm = ModifiedGaussianModel()
        mgm.load_spectrum(csv_file=file_path)
        
        print(f"Fitting {N_BANDS} absorption bands...")
        success = mgm.fit_spectrum(n_bands=N_BANDS)
        
        if success:
            mgm.plot_results(figsize=FIGURE_SIZE)
            return mgm
        else:
            print("Fitting failed. Try adjusting N_BANDS or check data quality.")
            return None
            
    except Exception as e:
        print(f"Error loading spectrum: {e}")
        print("Make sure your CSV has two columns: [wavelength, reflectance]")
        return None

def example_usage():
    """
    Example showing how to use the MGM class with synthetic data
    """
    print("Running synthetic example...")
    
    # Create synthetic UV-VIS spectrum (350-1000nm range)
    wavelength = np.linspace(350, 1000, 400)
    
    # Synthetic spectrum with absorption bands typical of UV-VIS range
    # Simulate continuum with slight slope
    continuum = 0.4 + 0.0002 * wavelength
    
    # Add absorption bands typical of Fe3+ and other transitions in UV-VIS
    band1 = -0.12 * np.exp(-((wavelength - 450) / 40)**2)   # Blue region absorption
    band2 = -0.18 * np.exp(-((wavelength - 650) / 60)**2)   # Red region absorption  
    band3 = -0.08 * np.exp(-((wavelength - 850) / 80)**2)   # Near-IR absorption
    
    # Add realistic noise
    noise = np.random.normal(0, 0.003, len(wavelength))
    
    reflectance = continuum + band1 + band2 + band3 + noise
    reflectance = np.clip(reflectance, 0.05, 0.95)  # Keep physically reasonable for minerals
    
    # Apply MGM
    mgm = ModifiedGaussianModel()
    mgm.load_spectrum(wavelength=wavelength, reflectance=reflectance)
    
    success = mgm.fit_spectrum(n_bands=N_BANDS)
    if success:
        mgm.plot_results(figsize=FIGURE_SIZE)
    
    return mgm


if __name__ == "__main__":
    print("MGM Spectral Analysis Tool")
    print("=" * 50)
    
    # Option 1: Run synthetic example (good for testing)
    print("Running synthetic example first...")
    example_result = example_usage()
    
    # Option 2: Try to analyze real data (uncomment next 2 lines if you have a real file)
    # print("\nNow analyzing real spectrum...")
    # real_result = analyze_real_spectrum()
    
    print("\nAnalysis complete! Plots should be displayed above.")