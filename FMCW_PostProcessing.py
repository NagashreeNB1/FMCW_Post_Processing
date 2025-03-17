import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import eig

class FMCWProcessor:
    def __init__(self, adc_data, sample_rate, chirp_rate, num_antennas, element_spacing, wavelength):
        """
        Initialize the FMCWProcessor with raw ADC data and array parameters.
        adc_data: 4D array with dimensions (rx, chirps, samples)
        sample_rate: Sampling rate in Hz
        chirp_rate: Chirp repetition rate in Hz
        num_antennas: Number of ULA elements (Rx antennas)
        element_spacing: Spacing between array elements (meters)
        wavelength: Wavelength of the signal (meters)
        """
        self.adc_data = adc_data
        self.sample_rate = sample_rate
        self.chirp_rate = chirp_rate
        self.num_antennas = num_antennas
        self.element_spacing = element_spacing
        self.wavelength = wavelength
        self.tx_data = None  # To store separated Tx contributions

    @staticmethod
    def generate_synthetic_data(num_tx, num_rx, num_chirps, num_samples, sampling_rate, frequency_slope, wavelength, target_params):
        """
        Generate synthetic ADC data for a radar system.
        target_params: List of dictionaries with target properties (range, velocity, AoA)
        """
        adc_data = np.zeros((num_rx, num_chirps, num_samples), dtype=complex)

        for target in target_params:
            target_range = target['range']
            target_velocity = target['velocity']
            target_aoa = target['aoa']
            target_rcs = target.get('rcs', 1)

            for chirp_idx in range(num_chirps):
                # Simulate range phase shift
                round_trip_time = 2 * target_range / 3e8  # seconds
                range_phase = 2 * np.pi * frequency_slope * round_trip_time * np.arange(num_samples) / sampling_rate

                # Simulate Doppler shift
                doppler_shift = 2 * target_velocity / wavelength  # Hz
                doppler_phase = 2 * np.pi * doppler_shift * chirp_idx * num_samples / sampling_rate

                # Simulate AoA phase shift for each Rx
                for rx_idx in range(num_rx):
                    aoa_phase = 2 * np.pi * wavelength / 2 * rx_idx * np.sin(np.radians(target_aoa)) / wavelength
                    adc_data[rx_idx, chirp_idx, :] += target_rcs * np.exp(1j * (range_phase + doppler_phase + aoa_phase))

        # Add noise to simulate real-world conditions
        noise_power = 0.1
        adc_data += np.sqrt(noise_power / 2) * (np.random.randn(*adc_data.shape) + 1j * np.random.randn(*adc_data.shape))
        return adc_data

    def separate_tx_data(self, orthogonal_codes):
        """
        Separate the contributions from each Tx using orthogonal coding.
        orthogonal_codes: Array of orthogonal codes (e.g., frequency or phase shifts) used for each Tx
        """
        num_rx, num_chirps, num_samples = self.adc_data.shape
        num_tx = len(orthogonal_codes)

        self.tx_data = np.zeros((num_tx, num_rx, num_chirps, num_samples), dtype=complex)

        for tx in range(num_tx):
            code = orthogonal_codes[tx]
            self.tx_data[tx] = self.adc_data * np.exp(-1j * 2 * np.pi * code * np.arange(num_samples) / num_samples)

    def compensate_ddm(self, phase_offsets, time_offsets, gain_factors):
        """
        Compensate for Doppler Division Multiplexing (DDM) phase, timing, and channel gain offsets.
        """
        for tx in range(self.adc_data.shape[0]):
            self.adc_data[tx] *= np.exp(-1j * phase_offsets[tx])
            self.adc_data[tx] = np.roll(self.adc_data[tx], -time_offsets[tx], axis=-1)
            self.adc_data[tx] /= gain_factors[tx]

    def range_fft(self):
        """Compute range FFT for each chirp."""
        range_data = np.fft.fft(self.adc_data, axis=2)
        return np.abs(range_data)

    def doppler_fft(self):
        """Compute Doppler FFT after range compression."""
        range_compressed = self.range_fft()
        doppler_data = np.fft.fft(range_compressed, axis=1)
        return np.abs(doppler_data)

    def plot_range(self):
        """Plot the average range spectrum."""
        range_data = self.range_fft().mean(axis=(0, 1))
        plt.plot(range_data)
        plt.title("Range Spectrum")
        plt.xlabel("Range Bins")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_doppler(self):
        """Plot the average Doppler spectrum."""
        doppler_data = self.doppler_fft().mean(axis=(0, 2))
        plt.plot(doppler_data)
        plt.title("Doppler Spectrum")
        plt.xlabel("Doppler Bins")
        plt.ylabel("Amplitude")
        plt.show()


if __name__ == '__main__':
    # Define radar and target parameters
    num_tx = 4
    num_rx = 4
    num_chirps = 128
    num_samples = 1024
    sampling_rate = 40e6
    frequency_slope = 200e6 / (num_samples / sampling_rate)
    wavelength = 3e8 / 77e9  # For a 77 GHz radar

    # Define target parameters
    target_params = [
        {"range": 10, "velocity": 20 / 3.6, "aoa": 10, "rcs": 1}  # Range 10m, Velocity 20 km/h, AoA 10 degrees
    ]

    # Generate synthetic data
    adc_data = FMCWProcessor.generate_synthetic_data(
        num_tx=num_tx,
        num_rx=num_rx,
        num_chirps=num_chirps,
        num_samples=num_samples,
        sampling_rate=sampling_rate,
        frequency_slope=frequency_slope,
        wavelength=wavelength,
        target_params=target_params
    )

    # Initialize processor
    processor = FMCWProcessor(
        adc_data=adc_data,
        sample_rate=sampling_rate,
        chirp_rate=1e3,
        num_antennas=num_rx,
        element_spacing=wavelength / 2,
        wavelength=wavelength
    )

    # Perform range and Doppler FFT
    processor.plot_range()
    processor.plot_doppler()

    # Example Tx separation using orthogonal codes
    orthogonal_codes = [0, 1, 2, 3]
    processor.separate_tx_data(orthogonal_codes)


