import numpy as np
import heapq
from collections import Counter
import folium
import math
from scipy.linalg import hadamard
import configparser

#Parsing settings from settings.conf
config = configparser.ConfigParser()
config.read("settings.conf")

CENTER_LAT = config.getfloat('Settings','center_lat')
CENTER_LON = config.getfloat('Settings','center_lon')
NUM_SENSORS = config.getint('Settings','num_sensors')
BANDWIDTH = config.getfloat('Settings','bandwidth_bps')
LATENCY = config.getfloat('Settings','latency_ms')
BEP = config.getfloat('Settings','bit_error_prob')
CODE_LEN = config.getint('Settings','code_len')
SPREAD = config.getfloat('Settings','spread')
RECEIVER_LAT = config.getfloat('Settings','reciever_lat')
RECEIVER_LON = config.getfloat('Settings','reciever_lon')


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_sensors(num_sensors, center_lat, center_lon, spread):
   
    lats = center_lat + np.random.uniform(-spread, spread, num_sensors)
    lon_spread = spread / math.cos(math.radians(center_lat))  
    lons = center_lon + np.random.uniform(-lon_spread, lon_spread, num_sensors)
    moisture = np.random.uniform(0, 100, num_sensors)
    return lats, lons, moisture

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol, self.freq = symbol, freq
        self.left = self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = Counter(data)
    heap = [(f, id(HuffmanNode()), HuffmanNode(sym, f)) for sym, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        f, _, node = heap[0]
        root = HuffmanNode(None, f*2)
        root.left = node
        heap = [(root.freq, id(root), root)]
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent = HuffmanNode(None, f1+f2)
        parent.left, parent.right = n1, n2
        heapq.heappush(heap, (parent.freq, id(parent), parent))
    return heap[0][2]

def make_codebook(root):
    codes = {}
    def dfs(node, prefix=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
        dfs(node.left, prefix + "0")
        dfs(node.right, prefix + "1")
    dfs(root)
    return codes

def huffman_encode(symbol, codebook):
    return codebook[symbol]

def huffman_decode(bitstr, root):
    out, node = [], root
    for b in bitstr:
        node = node.left if b == "0" else node.right
        if node is None:
            return []
        if node.symbol is not None:
            out.append(node.symbol)
            node = root
    return out

G = np.array([
    [1,1,0,1], [1,0,1,1], [1,0,0,0], [0,1,1,1],
    [0,1,0,0], [0,0,1,0], [0,0,0,1]
], dtype=int)
H = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1]
], dtype=int)

def hamming_encode(bits4):
    d = np.array(bits4, dtype=int)
    cw = G.dot(d) % 2
    return cw.tolist()

def hamming_decode(cw7):
    r = np.array(cw7, dtype=int)
    syndrome = H.dot(r) % 2
    idx = syndrome[0]*4 + syndrome[1]*2 + syndrome[2]
    if idx != 0 and idx <= 7:
        r[idx-1] ^= 1
    return [r[2], r[4], r[5], r[6]]

def generate_hadamard_codes(num_sensors, code_len=128):
    Hm = hadamard(code_len)
    return Hm[1:num_sensors+1]

def spread_bits(bitstream, code):
    bpsk = np.array(bitstream)*2 - 1
    return np.repeat(bpsk, len(code)) * np.tile(code, len(bpsk))

def despread(chips, code):
    L = len(code)
    mat = chips.reshape(-1, L)
    corr = mat.dot(code)
    return (corr > 0).astype(int).tolist()

np.random.seed(0)
lats, lons, moist = generate_sensors(NUM_SENSORS, CENTER_LAT, CENTER_LON, SPREAD)
moist_int = np.round(moist).astype(int)

huff_root = build_huffman_tree(moist_int.tolist())
codebook = make_codebook(huff_root)
id_bits = math.ceil(math.log2(NUM_SENSORS))

dp_codes = generate_hadamard_codes(NUM_SENSORS, CODE_LEN)

streams = []
for sid, val in enumerate(moist_int, start=1):
    id_str = format(sid-1, f'0{id_bits}b')
    hstr = huffman_encode(val, codebook)
    bits = [int(b) for b in id_str + hstr]
    pad = (-len(bits)) % 4
    bits += [0]*pad
    coded = []
    for i in range(0, len(bits), 4):
        coded.extend(hamming_encode(bits[i:i+4]))
    streams.append(coded)

max_len = max(len(s) for s in streams)
padded = [s + [0]*(max_len-len(s)) for s in streams]
spreaded = [spread_bits(s, dp_codes[i]) for i, s in enumerate(padded)]
composite = sum(spreaded)

mask = np.where(np.random.rand(*composite.shape) < BEP, -1, 1)
rx = composite * mask

decided = []
for i in range(NUM_SENSORS):
    chips = rx[:len(padded[i])*CODE_LEN]
    bits_hat = despread(chips, dp_codes[i])[:len(streams[i])]
    rec_stream = []
    for j in range(0, len(bits_hat), 7):
        rec_stream.extend(hamming_decode(bits_hat[j:j+7]))
    rec_stream = rec_stream[:len(streams[i])]
    rid = int(''.join(map(str, rec_stream[:id_bits])), 2)
    hbs = ''.join(map(str, rec_stream[id_bits:]))
    sym = huffman_decode(hbs, huff_root)
    rec_val = sym[0] if len(sym)==1 else None
    decided.append((rid, rec_val))

tx_time = len(composite)/BANDWIDTH
total_delay_ms = LATENCY + tx_time*1000
print(f"CDMA error rate: {np.mean([rv!=mi for (_,rv),mi in zip(decided,moist_int)]):.3f}, delay: {total_delay_ms:.1f} ms")

map_center = [CENTER_LAT, CENTER_LON]
folium_map = folium.Map(location=map_center, zoom_start=15)
folium.TileLayer('Stamen Terrain', name='Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0').add_to(folium_map)

folium.CircleMarker(
    location=[RECEIVER_LAT, RECEIVER_LON],
    radius=8,
    color='red',
    fill=True,
    fill_color='red',
    fill_opacity=0.9,
    popup=f"Receiver Location ({RECEIVER_LAT:.5f}, {RECEIVER_LON:.5f})"
).add_to(folium_map)

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

folium_map.save('soil_moisture_cdma_map.html')
print("Map saved as soil_moisture_cdma_map.html")