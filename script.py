import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(42)

N = 1500  # number of samples

packet_types = [
    "control_syn_ack",
    "dns_query",
    "web_small_obj",
    "web_large_obj",
    "voip_rtp",
    "gaming_udp",
    "video_chunk",
    "file_transfer_tcp",
    "iot_mqtt",
]

access_types = ["4G", "5G", "FixedWireless", "Fiber", "LEO_Satellite"]

# ---------- Helper Functions ----------
def sample_size_bytes(pkt):
    ranges = {
        "control_syn_ack": (40, 80),
        "dns_query": (60, 120),
        "web_small_obj": (2_000, 15_000),
        "web_large_obj": (50_000, 1_500_000),
        "voip_rtp": (60, 200),
        "gaming_udp": (50, 150),
        "video_chunk": (250_000, 5_000_000),
        "file_transfer_tcp": (1_000_000, 30_000_000),
        "iot_mqtt": (40, 400),
    }
    return np.random.randint(*ranges[pkt])

def sample_base_rtt_ms(access):
    ranges = {
        "5G": (8, 25),
        "4G": (20, 70),
        "FixedWireless": (15, 40),
        "Fiber": (5, 15),
        "LEO_Satellite": (35, 70),
    }
    return np.random.uniform(*ranges[access])

def sample_throughput_mbps(access, qos):
    base = {
        "5G": (50, 600),
        "4G": (5, 100),
        "FixedWireless": (20, 150),
        "Fiber": (100, 1000),
        "LEO_Satellite": (10, 150),
    }[access]
    bias = {
        "ultra_reliable_low_latency": 1.1,
        "voice": 1.0,
        "video": 0.9,
        "best_effort": 0.8,
    }[qos]
    lo, hi = base
    s = np.random.uniform(lo, hi) * bias
    return max(s, 1.0)

def sample_congestion_level():
    return np.clip(np.random.beta(2, 4), 0, 1)

def sample_jitter_ms(access, congestion):
    base = {
        "5G": (0.2, 3),
        "4G": (0.5, 8),
        "FixedWireless": (0.5, 5),
        "Fiber": (0.1, 1.5),
        "LEO_Satellite": (1, 12),
    }[access]
    return np.random.uniform(*base) * (0.6 + 1.2 * congestion)

def sample_loss_rate(access, congestion):
    base = {
        "5G": (0.0, 0.3),
        "4G": (0.0, 1.5),
        "FixedWireless": (0.0, 1.0),
        "Fiber": (0.0, 0.2),
        "LEO_Satellite": (0.0, 2.0),
    }[access]
    lo, hi = base
    loss = np.random.uniform(lo, hi) * (0.5 + 1.5 * congestion)
    return np.clip(loss, 0, 10)

def queue_delay_ms(congestion, qos):
    base = 100 * congestion**2
    priority = {
        "ultra_reliable_low_latency": 0.2,
        "voice": 0.4,
        "video": 0.8,
        "best_effort": 1.0,
    }[qos]
    return base * priority

def processing_delay_ms(pkt_type):
    base = {
        "control_syn_ack": (0.2, 1.0),
        "dns_query": (0.3, 2.0),
        "web_small_obj": (0.5, 3.0),
        "web_large_obj": (1.0, 5.0),
        "voip_rtp": (0.2, 1.0),
        "gaming_udp": (0.2, 1.2),
        "video_chunk": (0.8, 3.0),
        "file_transfer_tcp": (1.0, 4.0),
        "iot_mqtt": (0.2, 1.0),
    }[pkt_type]
    return np.random.uniform(*base)

def retransmission_penalty_ms(loss_rate_percent, protocol):
    loss = loss_rate_percent / 100.0
    if protocol == "TCP":
        return 2000 * (loss**1.2)
    else:
        return 400 * (loss**1.1)

def protocol_for(pkt):
    return {
        "control_syn_ack": "TCP",
        "dns_query": "UDP",
        "web_small_obj": "TCP",
        "web_large_obj": "TCP",
        "voip_rtp": "UDP",
        "gaming_udp": "UDP",
        "video_chunk": "TCP",
        "file_transfer_tcp": "TCP",
        "iot_mqtt": "TCP",
    }[pkt]

def qos_for(pkt):
    return {
        "control_syn_ack": "best_effort",
        "dns_query": "best_effort",
        "web_small_obj": "best_effort",
        "web_large_obj": "best_effort",
        "voip_rtp": "voice",
        "gaming_udp": "ultra_reliable_low_latency",
        "video_chunk": "video",
        "file_transfer_tcp": "best_effort",
        "iot_mqtt": "ultra_reliable_low_latency",
    }[pkt]

def backhaul_for(access):
    if access in ["Fiber"]:
        return "fiber"
    if access in ["4G", "5G"]:
        return np.random.choice(["fiber", "microwave", "mmwave", "mixed"], p=[0.6, 0.2, 0.1, 0.1])
    if access in ["FixedWireless"]:
        return np.random.choice(["microwave", "mmwave", "mixed"], p=[0.5, 0.3, 0.2])
    if access == "LEO_Satellite":
        return "mixed"
    return "mixed"

def path_hops_for(access):
    base = {
        "5G": (8, 16),
        "4G": (10, 20),
        "FixedWireless": (8, 18),
        "Fiber": (6, 14),
        "LEO_Satellite": (12, 22),
    }[access]
    return np.random.randint(*base)

def transmission_delay_ms(size_bytes, throughput_mbps):
    bits = size_bytes * 8
    seconds = bits / (throughput_mbps * 1_000_000)
    return seconds * 1000

# ---------- Dataset Generation ----------
rows = []
for _ in range(N):
    pkt = np.random.choice(packet_types)
    access = np.random.choice(access_types, p=[0.35, 0.25, 0.1, 0.2, 0.1])
    qos = qos_for(pkt)
    proto = protocol_for(pkt)
    size_b = sample_size_bytes(pkt)
    base_rtt = sample_base_rtt_ms(access)
    hops = path_hops_for(access)
    backhaul = backhaul_for(access)
    throughput = sample_throughput_mbps(access, qos)
    congestion = sample_congestion_level()
    jitter = sample_jitter_ms(access, congestion)
    q_delay = queue_delay_ms(congestion, qos)
    proc_delay = processing_delay_ms(pkt)
    loss = sample_loss_rate(access, congestion)
    retrans_ms = retransmission_penalty_ms(loss, proto)
    tx_delay = transmission_delay_ms(size_b, throughput)

    one_way_base = base_rtt / 2.0
    final_ms = one_way_base + tx_delay + q_delay + proc_delay + jitter + retrans_ms

    rows.append({
        "timestamp": datetime.utcnow().isoformat(),
        "packet_type": pkt,
        "protocol": proto,
        "qos_class": qos,
        "access_type": access,
        "backhaul_type": backhaul,
        "path_hops": hops,
        "size_bytes": size_b,
        "throughput_mbps": throughput,
        "base_rtt_ms": base_rtt,
        "jitter_ms": jitter,
        "congestion_level_0to1": congestion,
        "loss_rate_percent": loss,
        "queue_delay_ms": q_delay,
        "processing_delay_ms": proc_delay,
        "tx_delay_ms": tx_delay,
        "retransmission_penalty_ms": retrans_ms,
        "final_one_way_time_ms": final_ms,
    })

df = pd.DataFrame(rows)
df.to_csv("synthetic_internet_to_bs_latency_dataset.csv", index=False)
print("Dataset saved as synthetic_internet_to_bs_latency_dataset.csv")
print(df.head())

