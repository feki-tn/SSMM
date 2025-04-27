# CDMA Soil Moisture Sensor Simulation

This project simulates a **Code Division Multiple Access (CDMA)** system for transmitting soil moisture data from multiple sensors to a receiver. It incorporates **Huffman coding** for data compression, **Hamming codes** for error correction, and **Hadamard codes** for signal spreading. The simulation also visualizes the sensor locations and their data on an interactive map using Folium.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/feki-tn/SSMM.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd SSMM
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The simulation parameters are defined in the `settings.conf` file. You can modify this file to change the following parameters:
- Center coordinates for sensor placement (`center_lat`, `center_lon`)
- Number of sensors (`num_sensors`)
- Bandwidth in bits per second (`bandwidth_bps`)
- Fixed latency in milliseconds (`latency_ms`)
- Bit error probability (`bit_error_prob`)
- Length of Hadamard codes for spreading (`code_len`)
- Spread range for sensor positions (`spread`)
- Receiver coordinates (`reciever_lat`, `reciever_lon`)

## Usage

To run the simulation, execute the following command:

```bash
python ssmm.py
```

This will:
- Simulate the CDMA transmission of soil moisture data.
- Calculate the error rate and total delay.
- Generate an interactive map visualizing the sensor locations, their original and recovered moisture values, and the delay.

The simulation will output the CDMA error rate and the total delay in the console. It will also save a map as `soil_moisture_cdma_map.html`, which you can open in a web browser to visualize the sensor data.

## Technologies Used

- Python 3
- NumPy
- SciPy
- Folium
- ConfigParser

## Contributing

If you find any issues or have suggestions for improvements, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.