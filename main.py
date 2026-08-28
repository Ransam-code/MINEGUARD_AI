"""
MINEGUARD AI - Software Processing Layer
=========================================
Ingests serial gateway data (mock), maintains per-node sliding windows,
performs Z-score anomaly detection, and correlates hardware + software
risk into a final safety verdict.
"""

import sys
import io
import random
import time

# Force UTF-8 on Windows so box-drawing and symbols render correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
WINDOW_SIZE = 10          # sliding-window depth per sensor
Z_THRESHOLD = 2.0         # flag anomaly when |z| exceeds this
NODES = ["NODE_01", "NODE_02", "NODE_03"]
RISK_LEVELS = ["SAFE", "WARNING", "CRITICAL"]

# Sensor ranges used by the mock generator
SENSOR_RANGES = {
    "Tilt": (0.0, 45.0),       # degrees
    "Vib":  (200, 5000),        # arbitrary vibration units
    "Dist": (10.0, 100.0),      # cm
}

# Occasionally inject a spike so the anomaly detector has something to find
SPIKE_PROBABILITY = 0.12


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class SensorWindow:
    """Per-sensor sliding window with rolling statistics."""
    name: str
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))

    def push(self, value: float) -> Tuple[float, float, bool]:
        """
        Append a reading and return (mean, stdev, is_anomaly).
        An anomaly is flagged when |z-score| > Z_THRESHOLD and the
        window has at least 3 samples (so stdev is meaningful).
        """
        self.window.append(value)

        mean = statistics.mean(self.window)

        if len(self.window) < 3:
            return mean, 0.0, False

        stdev = statistics.pstdev(self.window)      # population stdev of window

        if stdev == 0:
            return mean, 0.0, False

        z_score = (value - mean) / stdev
        is_anomaly = abs(z_score) > Z_THRESHOLD
        return mean, stdev, is_anomaly


@dataclass
class NodeState:
    """Holds the three sensor windows for a single node."""
    tilt: SensorWindow = field(default_factory=lambda: SensorWindow("Tilt"))
    vib:  SensorWindow = field(default_factory=lambda: SensorWindow("Vib"))
    dist: SensorWindow = field(default_factory=lambda: SensorWindow("Dist"))


# ──────────────────────────────────────────────
# Mock serial data generator
# ──────────────────────────────────────────────
def get_serial_data() -> str:
    """
    Simulate a line arriving from the LoRa gateway.

    Format:
        GATEWAY RECEIVED -> NODE_01 | Tilt: 12.50 | Vib: 1800 | Dist: 45.20 | Risk: WARNING
    """
    node = random.choice(NODES)

    tilt_lo, tilt_hi = SENSOR_RANGES["Tilt"]
    vib_lo,  vib_hi  = SENSOR_RANGES["Vib"]
    dist_lo, dist_hi = SENSOR_RANGES["Dist"]

    tilt = round(random.uniform(tilt_lo, tilt_hi), 2)
    vib  = random.randint(vib_lo, vib_hi)
    dist = round(random.uniform(dist_lo, dist_hi), 2)

    # Inject occasional spikes to exercise the anomaly detector
    if random.random() < SPIKE_PROBABILITY:
        sensor = random.choice(["tilt", "vib", "dist"])
        if sensor == "tilt":
            tilt = round(tilt * random.uniform(2.5, 4.0), 2)
        elif sensor == "vib":
            vib = int(vib * random.uniform(3.0, 5.0))
        else:
            dist = round(dist * random.uniform(2.5, 4.0), 2)

    # Hardware risk heuristic (mirrors what the embedded firmware might send)
    if tilt > 35 or vib > 4000 or dist > 85:
        risk = "CRITICAL"
    elif tilt > 20 or vib > 2500 or dist > 60:
        risk = "WARNING"
    else:
        risk = "SAFE"

    return (
        f"GATEWAY RECEIVED -> {node} "
        f"| Tilt: {tilt:.2f} "
        f"| Vib: {vib} "
        f"| Dist: {dist:.2f} "
        f"| Risk: {risk}"
    )


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────
def parse_serial_data(raw: str) -> Dict:
    """
    Parse a gateway string into a structured dict.

    Returns:
        {
            "node": "NODE_01",
            "Tilt": 12.5,
            "Vib": 1800.0,
            "Dist": 45.2,
            "hw_risk": "WARNING",
        }
    """
    # Split on the arrow to isolate the payload
    _, payload = raw.split("->")
    parts = [p.strip() for p in payload.split("|")]

    node    = parts[0]
    tilt    = float(parts[1].split(":")[1])
    vib     = float(parts[2].split(":")[1])
    dist    = float(parts[3].split(":")[1])
    hw_risk = parts[4].split(":")[1].strip()

    return {
        "node": node,
        "Tilt": tilt,
        "Vib": vib,
        "Dist": dist,
        "hw_risk": hw_risk,
    }


# ──────────────────────────────────────────────
# Risk correlation
# ──────────────────────────────────────────────
def correlate_risk(hw_risk: str, anomalies: Dict[str, bool]) -> str:
    """
    Final risk verdict.

    Escalate to CRITICAL if:
      • the hardware already says CRITICAL, **or**
      • any software anomaly was detected.
    Otherwise, pass the hardware risk through unchanged.
    """
    sw_anomaly = any(anomalies.values())

    if hw_risk == "CRITICAL" or sw_anomaly:
        return "CRITICAL"
    return hw_risk


# ──────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────
ANSI_RESET  = "\033[0m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_CYAN   = "\033[96m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"

RISK_COLOR = {
    "SAFE":     ANSI_GREEN,
    "WARNING":  ANSI_YELLOW,
    "CRITICAL": ANSI_RED,
}


def print_banner():
    """Print a startup banner."""
    print(f"""
{ANSI_CYAN}{ANSI_BOLD}╔══════════════════════════════════════════════════════════════╗
║             MINEGUARD AI  —  Software Processing Layer       ║
║          Sliding-Window Z-Score Anomaly Detection Engine      ║
╚══════════════════════════════════════════════════════════════╝{ANSI_RESET}
{ANSI_DIM}  Window Size : {WINDOW_SIZE}   |   Z-Score Threshold : {Z_THRESHOLD}
  Nodes       : {', '.join(NODES)}{ANSI_RESET}
""")


def format_report(
    reading: Dict,
    stats: Dict[str, Tuple[float, float, bool]],
    final_risk: str,
) -> str:
    """Build a human-readable report block for one reading."""

    node    = reading["node"]
    hw_risk = reading["hw_risk"]
    rc      = RISK_COLOR.get(final_risk, ANSI_RESET)

    lines = [
        f"{ANSI_DIM}{'─' * 62}{ANSI_RESET}",
        f"{ANSI_BOLD}  Node: {node}{ANSI_RESET}",
        f"  {'Sensor':<8} {'Value':>10} {'Mean':>10} {'StDev':>10}  {'Status'}",
    ]

    for sensor in ("Tilt", "Vib", "Dist"):
        val = reading[sensor]
        mean, stdev, anomaly = stats[sensor]
        flag = f"{ANSI_RED}✦ ANOMALY{ANSI_RESET}" if anomaly else f"{ANSI_GREEN}  OK{ANSI_RESET}"
        lines.append(
            f"  {sensor:<8} {val:>10.2f} {mean:>10.2f} {stdev:>10.2f}  {flag}"
        )

    hw_color = RISK_COLOR.get(hw_risk, ANSI_RESET)
    lines.append(f"\n  HW Risk : {hw_color}{hw_risk}{ANSI_RESET}")
    lines.append(f"  {ANSI_BOLD}Final   : {rc}{final_risk}{ANSI_RESET}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────
def main():
    print_banner()

    # Per-node state: { "NODE_01": NodeState, ... }
    nodes: Dict[str, NodeState] = {}

    try:
        while True:
            raw = get_serial_data()
            print(f"\n{ANSI_DIM}  ◂ {raw}{ANSI_RESET}")

            reading = parse_serial_data(raw)
            node_id = reading["node"]

            # Lazily initialise node state
            if node_id not in nodes:
                nodes[node_id] = NodeState()

            state = nodes[node_id]

            # Push readings into sliding windows and collect stats
            stats: Dict[str, Tuple[float, float, bool]] = {}
            stats["Tilt"] = state.tilt.push(reading["Tilt"])
            stats["Vib"]  = state.vib.push(reading["Vib"])
            stats["Dist"] = state.dist.push(reading["Dist"])

            # Anomaly flags for correlation
            anomalies = {
                sensor: stats[sensor][2]   # the is_anomaly boolean
                for sensor in ("Tilt", "Vib", "Dist")
            }

            final_risk = correlate_risk(reading["hw_risk"], anomalies)

            print(format_report(reading, stats, final_risk))

            time.sleep(1)      # simulate 1 Hz sample rate

    except KeyboardInterrupt:
        print(f"\n{ANSI_CYAN}  MINEGUARD processing stopped.{ANSI_RESET}\n")


if __name__ == "__main__":
    main()
