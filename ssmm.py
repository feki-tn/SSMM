import numpy as np
import heapq
from collections import Counter
import folium
import math
from scipy.linalg import hadamard
import configparser

# Load configuration settings from a file
config = configparser.ConfigParser()
config.read("settings.conf")

# Extract simulation parameters from the config file
CENTER_LAT = config.getfloat('Settings', 'center_lat')      # Center latitude for sensor placement
CENTER_LON = config.getfloat('Settings', 'center_lon')      # Center longitude for sensor placement
NUM_SENSORS = config.getint('Settings', 'num_sensors')      # Number of sensors in the simulation
BANDWIDTH = config.getfloat('Settings', 'bandwidth_bps')    # Bandwidth in bits per second
LATENCY = config.getfloat('Settings', 'latency_ms')         # Fixed latency in milliseconds
BEP = config.getfloat('Settings', 'bit_error_prob')         # Bit error probability
CODE_LEN = config.getint('Settings', 'code_len')            # Length of Hadamard codes for spreading
SPREAD = config.getfloat('Settings', 'spread')              # Spread range for sensor positions
RECEIVER_LAT = config.getfloat('Settings', 'reciever_lat')  # Receiver latitude
RECEIVER_LON = config.getfloat('Settings', 'reciever_lon')  # Receiver longitude

# Calculate distance between two points on Earth using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)  # Difference in latitude, converted to radians
    dlon = math.radians(lon2 - lon1)  # Difference in longitude, converted to radians
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # Great-circle distance
    return R * c  

# Generate random sensor positions and moisture values
def generate_sensors(num_sensors, center_lat, center_lon, spread):
    lats = center_lat + np.random.uniform(-spread, spread, num_sensors)  # Random latitudes around center
    lon_spread = spread / math.cos(math.radians(center_lat))  # Adjust longitude spread for latitude
    lons = center_lon + np.random.uniform(-lon_spread, lon_spread, num_sensors)  # Random longitudes
    moisture = np.random.uniform(0, 100, num_sensors)  # Random moisture values between 0 and 100
    return lats, lons, moisture

# Define a node class for Huffman coding tree
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol  # Symbol (e.g., moisture value)
        self.freq = freq      # Frequency of the symbol
        self.left = None      # Left child node
        self.right = None     # Right child node
    def __lt__(self, other):
        return self.freq < other.freq  # Comparison for priority queue

# Build a Huffman tree based on symbol frequencies
def build_huffman_tree(data):
    freq = Counter(data)  # Count frequency of each symbol
    heap = [(f, id(HuffmanNode()), HuffmanNode(sym, f)) for sym, f in freq.items()]  # Create initial nodes
    heapq.heapify(heap)  # Convert to a min-heap
    if len(heap) == 1:  # Special case: only one symbol
        f, _, node = heap[0]
        root = HuffmanNode(None, f*2)
        root.left = node
        heap = [(root.freq, id(root), root)]
    while len(heap) > 1:  # Merge nodes until one tree remains
        f1, _, n1 = heapq.heappop(heap)  # Pop two nodes with smallest frequencies
        f2, _, n2 = heapq.heappop(heap)
        parent = HuffmanNode(None, f1 + f2)  # Create parent node
        parent.left, parent.right = n1, n2   # Assign children
        heapq.heappush(heap, (parent.freq, id(parent), parent))  # Push back to heap
    return heap[0][2]  # Return root of the tree

# Generate a codebook from the Huffman tree
def make_codebook(root):
    codes = {}  # Dictionary to store symbol-to-code mappings
    def dfs(node, prefix=""):  # Depth-first search to assign codes
        if node is None:
            return
        if node.symbol is not None:  # Leaf node
            codes[node.symbol] = prefix or "0"  # Assign code (default to "0" if empty)
        dfs(node.left, prefix + "0")   # Traverse left with "0"
        dfs(node.right, prefix + "1")  # Traverse right with "1"
    dfs(root)
    return codes

# Encode a symbol using the Huffman codebook
def huffman_encode(symbol, codebook):
    return codebook[symbol]  # Return the binary string for the symbol

# Decode a bitstring back to symbols using the Huffman tree
def huffman_decode(bitstr, root):
    out, node = [], root  # List for decoded symbols, start at root
    for b in bitstr:  # Traverse tree based on bits
        node = node.left if b == "0" else node.right
        if node is None:  # Invalid path
            return []
        if node.symbol is not None:  # Leaf node reached
            out.append(node.symbol)  # Add symbol to output
            node = root  # Reset to root
    return out

# Define matrices for Hamming(7,4) code
G = np.array([
    [1,1,0,1], [1,0,1,1], [1,0,0,0], [0,1,1,1],
    [0,1,0,0], [0,0,1,0], [0,0,0,1]
], dtype=int)  # Generator matrix
H = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1]
], dtype=int)  # Parity-check matrix

# Encode 4 bits into 7 bits using Hamming code
def hamming_encode(bits4):
    d = np.array(bits4, dtype=int)  # Convert input to numpy array
    cw = G.dot(d) % 2  # Matrix multiplication modulo 2
    return cw.tolist()  # Return as list

# Decode 7 bits back to 4 bits, correcting single-bit errors
def hamming_decode(cw7):
    r = np.array(cw7, dtype=int)  # Received codeword
    syndrome = H.dot(r) % 2  # Compute syndrome
    idx = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]  # Error position (0 if no error)
    if idx != 0 and idx <= 7:  # If error detected
        r[idx-1] ^= 1  # Flip the erroneous bit
    return [r[2], r[4], r[5], r[6]]  # Return data bits

# Generate Hadamard codes for CDMA spreading
def generate_hadamard_codes(num_sensors, code_len=128):
    Hm = hadamard(code_len)  # Generate Hadamard matrix
    return Hm[1:num_sensors+1]  # Return rows 1 to num_sensors (skip all-ones row)

# Spread a bitstream using a Hadamard code
def spread_bits(bitstream, code):
    bpsk = np.array(bitstream)*2 - 1  # Convert bits to BPSK (-1 or 1)
    return np.repeat(bpsk, len(code)) * np.tile(code, len(bpsk))  # Spread with code

# Despread a received signal to recover bits
def despread(chips, code):
    L = len(code)  # Length of the spreading code
    mat = chips.reshape(-1, L)  # Reshape into blocks
    corr = mat.dot(code)  # Correlate with code
    return (corr > 0).astype(int).tolist()  # Threshold to get bits

# Main simulation
np.random.seed(0)  # Set seed for reproducibility
lats, lons, moist = generate_sensors(NUM_SENSORS, CENTER_LAT, CENTER_LON, SPREAD)  # Generate sensor data
moist_int = np.round(moist).astype(int)  # Round moisture values to integers

huff_root = build_huffman_tree(moist_int.tolist())  # Build Huffman tree
codebook = make_codebook(huff_root)  # Create Huffman codebook

id_bits = math.ceil(math.log2(NUM_SENSORS))  # Bits needed for sensor IDs
dp_codes = generate_hadamard_codes(NUM_SENSORS, CODE_LEN)  # Generate spreading codes

# Encode each sensor's data
streams = []
for sid, val in enumerate(moist_int, start=1):
    id_str = format(sid-1, f'0{id_bits}b')  # Sensor ID as binary string
    hstr = huffman_encode(val, codebook)    # Huffman encode moisture value
    bits = [int(b) for b in id_str + hstr]  # Combine ID and moisture bits
    pad = (-len(bits)) % 4  # Padding to make length multiple of 4
    bits += [0]*pad
    coded = []  # Hamming encoded bits
    for i in range(0, len(bits), 4):  # Encode 4 bits at a time
        coded.extend(hamming_encode(bits[i:i+4]))
    streams.append(coded)

# Pad all streams to the same length
max_len = max(len(s) for s in streams)  # Find longest stream
padded = [s + [0]*(max_len - len(s)) for s in streams]  # Pad with zeros

# Spread each stream with its Hadamard code
spreaded = [spread_bits(s, dp_codes[i]) for i, s in enumerate(padded)]

# Create composite signal by summing all spread signals
composite = sum(spreaded)

# Simulate transmission errors
mask = np.where(np.random.rand(*composite.shape) < BEP, -1, 1)  # Random error mask
rx = composite * mask  # Apply errors to received signal

# Decode for each sensor
decided = []
for i in range(NUM_SENSORS):
    chips = rx[:len(padded[i])*CODE_LEN]  # Extract received chips
    bits_hat = despread(chips, dp_codes[i])[:len(streams[i])]  # Despread to bits
    rec_stream = []  # Decoded bits
    for j in range(0, len(bits_hat), 7):  # Decode 7-bit blocks
        rec_stream.extend(hamming_decode(bits_hat[j:j+7]))
    rec_stream = rec_stream[:len(streams[i])]  # Truncate to original length
    rid = int(''.join(map(str, rec_stream[:id_bits])), 2)  # Extract sensor ID
    hbs = ''.join(map(str, rec_stream[id_bits:]))  # Extract Huffman bits
    sym = huffman_decode(hbs, huff_root)  # Decode moisture value
    rec_val = sym[0] if len(sym) == 1 else None  # Get value or None if invalid
    decided.append((rid, rec_val))

# Calculate transmission time and total delay
tx_time = len(composite) / BANDWIDTH  # Transmission time in seconds
total_delay_ms = LATENCY + tx_time * 1000  # Total delay in milliseconds

# Print error rate and delay
print(f"CDMA error rate: {np.mean([rv!=mi for (_,rv),mi in zip(decided,moist_int)]):.3f}, delay: {total_delay_ms:.1f} ms")

# Create map visualization
map_center = [CENTER_LAT, CENTER_LON]
folium_map = folium.Map(location=map_center, zoom_start=15)  # Initialize map
folium.TileLayer('Stamen Terrain', name='Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0').add_to(folium_map)

# Add receiver marker
folium.CircleMarker(
    location=[RECEIVER_LAT, RECEIVER_LON],
    radius=8,
    color='red',
    fill=True,
    fill_color='red',
    fill_opacity=0.9,
    popup=f"Receiver Location ({RECEIVER_LAT:.5f}, {RECEIVER_LON:.5f})"
).add_to(folium_map)

# Add sensor markers
for idx, (rid, rec_val) in enumerate(decided):
    popup = (
        f"Sensor ID: {rid+1}<br>"
        f"Original: {moist[idx]:.1f}%<br>"
        f"Recovered: {rec_val if rec_val is not None else 'err'}%<br>"
        f"Delay: {total_delay_ms:.1f} ms"
    )
    folium.CircleMarker(
        location=[lats[idx], lons[idx]],
        radius=5,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=popup
    ).add_to(folium_map)

# Save the map
folium_map.save('soil_moisture_cdma_map.html')
print("Map saved as soil_moisture_cdma_map.html")