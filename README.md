# FMCW Post Processing

## Overview
The FMCW Post Processing project is designed to process and analyze data from Frequency Modulated Continuous Wave (FMCW) radar systems. It includes functionalities for generating synthetic radar data, separating transmit contributions, compensating for Doppler Division Multiplexing (DDM), and performing Fast Fourier Transform (FFT) operations for range and Doppler analysis.

## Features
- Generate synthetic ADC data for radar systems.
- Separate contributions from multiple transmitters using orthogonal coding.
- Compensate for phase, timing, and gain offsets in the received signals.
- Perform range and Doppler FFT to analyze target information.
- Visualize range and Doppler spectra.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd FMCW_PostProcessing
   ```
3. Install the required dependencies:
   ```
   pip install numpy matplotlib scipy
   ```

## Usage
To run the FMCW processing application, execute the following command:
```
python FMCW_PostProcessing.py
```

This will generate synthetic data, process it, and display the range and Doppler spectra.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.