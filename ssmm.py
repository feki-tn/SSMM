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
NUM_SENSORS = config.getfloat('Settings','num_sensors')
BANDWIDTH = config.getfloat('Settings','bandwidth_bps')
LATENCY = config.getfloat('Settings','latency_ms')
BEP = config.getfloat('Settings','bit_error_prob')
CODE_LEN = config.getfloat('Settings','code_len')
SPREAD = config.getfloat('Settings','spread')
RECEIVER_LAT = config.getfloat('Settings','reciever_lat')
RECEIVER_LON = config.getfloat('Settings','reciever_lon')