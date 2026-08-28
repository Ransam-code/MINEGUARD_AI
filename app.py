"""
MINEGUARD AI - Streamlit Real-Time Dashboard
=============================================
Single-file dashboard with mock serial ingestion, sliding-window
Z-score anomaly detection, and live Plotly visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import time
import statistics
from collections import deque
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MINEGUARD AI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
WINDOW_SIZE = 10
Z_THRESHOLD = 2.0
NODES = ["NODE_01", "NODE_02", "NODE_03"]
SENSORS = ["Tilt", "Vib", "Dist"]
MAX_CHART_POINTS = 50

SENSOR_RANGES = {
    "Tilt": (0.0, 45.0),
    "Vib":  (200, 5000),
    "Dist": (10.0, 100.0),
}

SENSOR_UNITS = {"Tilt": "deg", "Vib": "Hz", "Dist": "cm"}

SPIKE_PROBABILITY = 0.12

SENSOR_COLORS = {
    "Tilt": {"line": "#00d4ff", "fill": "rgba(0,212,255,0.1)"},
    "Vib":  {"line": "#ff9800", "fill": "rgba(255,152,0,0.1)"},
    "Dist": {"line": "#aa66ff", "fill": "rgba(170,102,255,0.1)"},
}

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(145deg, #0a0e17 0%, #0f1923 50%, #0a0e17 100%);
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}

/* ── Dashboard Header ── */
.dashboard-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(0,188,212,0.04) 100%);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
}
.dashboard-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #00e676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.dashboard-header .subtitle {
    color: #8b949e;
    font-size: 0.8rem;
    margin: 0.2rem 0 0 0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.9), rgba(13,17,23,0.95));
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0,212,255,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.metric-label {
    color: #8b949e;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
}
.metric-value.safe     { color: #00e676; }
.metric-value.warning  { color: #ffab00; }
.metric-value.critical { color: #ff1744; text-shadow: 0 0 20px rgba(255,23,68,0.4); }
.metric-value.cyan     { color: #00d4ff; }
.metric-value.green    { color: #00e676; }

/* ── Node Cards ── */
.node-card {
    background: linear-gradient(135deg, rgba(22,27,34,0.85), rgba(13,17,23,0.9));
    border: 1px solid rgba(48,54,61,0.5);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(10px);
}
.node-card.safe     { border-left: 3px solid #00e676; }
.node-card.warning  { border-left: 3px solid #ffab00; }
.node-card.critical { border-left: 3px solid #ff1744; box-shadow: inset 0 0 30px rgba(255,23,68,0.05); }

.node-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Status Dots ── */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.safe     { background: #00e676; box-shadow: 0 0 8px #00e676; }
.status-dot.warning  { background: #ffab00; box-shadow: 0 0 8px #ffab00; }
.status-dot.critical {
    background: #ff1744;
    box-shadow: 0 0 8px #ff1744;
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 8px #ff1744; }
    50%      { box-shadow: 0 0 20px #ff1744, 0 0 40px rgba(255,23,68,0.3); }
}

/* ── Sensor Rows ── */
.sensor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.78rem;
    color: #8b949e;
    border-bottom: 1px solid rgba(48,54,61,0.3);
}
.sensor-row:last-child { border-bottom: none; }
.sensor-name  { font-weight: 500; color: #c9d1d9; }
.sensor-val   { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.sensor-val.anomaly { color: #ff1744; }
.sensor-val.normal  { color: #00e676; }

/* ── Badges ── */
.anomaly-badge {
    display: inline-block;
    background: rgba(255,23,68,0.15);
    color: #ff1744;
    border: 1px solid rgba(255,23,68,0.3);
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.65rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.ok-badge {
    display: inline-block;
    background: rgba(0,230,118,0.1);
    color: #00e676;
    border: 1px solid rgba(0,230,118,0.2);
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.65rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Serial Log ── */
.serial-log {
    background: rgba(13,17,23,0.95);
    border: 1px solid rgba(48,54,61,0.5);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7ee787;
    max-height: 160px;
    overflow-y: auto;
    margin-top: 0.5rem;
}
.serial-line         { margin: 0.15rem 0; }
.serial-line .ts     { color: #484f58; }
.serial-line .arrow  { color: #00d4ff; }

/* ── Section Headers ── */
.section-header {
    color: #e6edf3;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.2rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(48,54,61,0.5);
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 3px; }
</style>"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def init_session_state():
    """Idempotent — safe to call on every rerun."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.tick = 0

        # Per-node sliding windows  { node: { sensor: deque } }
        st.session_state.windows = {
            n: {s: deque(maxlen=WINDOW_SIZE) for s in SENSORS}
            for n in NODES
        }

        # Per-node chart series  { node: { sensor: {times:[], values:[]} } }
        st.session_state.chart_data = {
            n: {s: {"times": [], "values": []} for s in SENSORS}
            for n in NODES
        }

        st.session_state.node_risk = {n: "SAFE" for n in NODES}
        st.session_state.node_anomalies = {
            n: {s: False for s in SENSORS} for n in NODES
        }
        st.session_state.latest_readings = {
            n: {s: 0.0 for s in SENSORS} for n in NODES
        }

        st.session_state.serial_log = deque(maxlen=30)
        st.session_state.anomaly_count = 0


init_session_state()


# ─────────────────────────────────────────────
# Data Source Functions
# ─────────────────────────────────────────────
def get_mock_serial_data(node=None):
    """
    Generate a mock gateway string.

    Returns a string like:
        GATEWAY RECEIVED -> NODE_01 | Tilt: 12.50 | Vib: 1800 | Dist: 45.20 | Risk: SAFE
    """
    if node is None:
        node = random.choice(NODES)

    tilt = round(random.uniform(*SENSOR_RANGES["Tilt"]), 2)
    vib  = random.randint(*SENSOR_RANGES["Vib"])
    dist = round(random.uniform(*SENSOR_RANGES["Dist"]), 2)

    # Occasional spike injection so the anomaly detector fires
    if random.random() < SPIKE_PROBABILITY:
        target = random.choice(SENSORS)
        if target == "Tilt":
            tilt = round(tilt * random.uniform(2.5, 4.0), 2)
        elif target == "Vib":
            vib = int(vib * random.uniform(3.0, 5.0))
        else:
            dist = round(dist * random.uniform(2.5, 4.0), 2)

    # Hardware-level risk heuristic
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


def get_real_serial_data(port="COM3", baudrate=9600):
    """Read one line from a physical serial port (pyserial)."""
    try:
        import serial  # noqa: F811 – only imported when toggle is on

        if "serial_conn" not in st.session_state:
            st.session_state.serial_conn = serial.Serial(
                port, baudrate, timeout=1,
            )

        line = (
            st.session_state.serial_conn
            .readline()
            .decode("utf-8", errors="replace")
            .strip()
        )
        return line if line else None
    except Exception:
        return None


def get_serial_data(node=None, use_real=False):
    """Unified data source selector."""
    if use_real:
        return get_real_serial_data()
    return get_mock_serial_data(node)


# ─────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────
def parse_serial_data(raw):
    """Parse a gateway string into a structured dict."""
    try:
        _, payload = raw.split("->")
        parts = [p.strip() for p in payload.split("|")]
        return {
            "node":    parts[0].strip(),
            "Tilt":    float(parts[1].split(":")[1]),
            "Vib":     float(parts[2].split(":")[1]),
            "Dist":    float(parts[3].split(":")[1]),
            "hw_risk": parts[4].split(":")[1].strip(),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# Z-Score Anomaly Detection
# ─────────────────────────────────────────────
def check_anomaly(window, value):
    """Return (mean, stdev, z_score, is_anomaly)."""
    mean = statistics.mean(window)
    if len(window) < 3:
        return mean, 0.0, 0.0, False
    stdev = statistics.pstdev(window)
    if stdev == 0:
        return mean, 0.0, 0.0, False
    z = (value - mean) / stdev
    return mean, stdev, z, abs(z) > Z_THRESHOLD


def correlate_risk(hw_risk, anomalies):
    """Escalate to CRITICAL if HW says so OR any software anomaly fires."""
    if hw_risk == "CRITICAL" or any(anomalies.values()):
        return "CRITICAL"
    return hw_risk


# ─────────────────────────────────────────────
# Chart Builder
# ─────────────────────────────────────────────
def build_node_chart(node):
    """Build a 3-row Plotly figure (Tilt / Vib / Dist) for one node."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f"Tilt ({SENSOR_UNITS['Tilt']})",
            f"Vibration ({SENSOR_UNITS['Vib']})",
            f"Distance ({SENSOR_UNITS['Dist']})",
        ],
    )

    for i, sensor in enumerate(SENSORS, 1):
        data   = st.session_state.chart_data[node][sensor]
        colors = SENSOR_COLORS[sensor]

        fig.add_trace(
            go.Scatter(
                x=data["times"],
                y=data["values"],
                mode="lines+markers",
                line=dict(color=colors["line"], width=2, shape="spline"),
                marker=dict(size=3, color=colors["line"]),
                fill="tozeroy",
                fillcolor=colors["fill"],
                showlegend=False,
            ),
            row=i, col=1,
        )

    fig.update_layout(
        height=380,
        margin=dict(l=45, r=15, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.7)",
        font=dict(family="Inter", size=10, color="#8b949e"),
    )

    for row in range(1, 4):
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(48,54,61,0.3)",
            showline=False, zeroline=False,
            tickfont=dict(size=9), row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(48,54,61,0.3)",
            showline=False, zeroline=False,
            tickfont=dict(size=9), row=row, col=1,
        )

    for ann in fig.layout.annotations:
        ann.font = dict(size=10, color="#c9d1d9", family="Inter")

    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### :gear: Configuration")
    use_real = st.toggle(
        "Use Real Serial (COM3)",
        value=False,
        key="use_real_serial",
    )

    st.divider()
    st.markdown(f"""
    ### :bar_chart: Parameters
    - **Window Size:** `{WINDOW_SIZE}` samples
    - **Z-Threshold:** `{Z_THRESHOLD}`
    - **Update Rate:** `1 Hz`
    - **Nodes:** `{len(NODES)}`
    """)

    st.divider()
    src = ":red_circle: COM3 Serial" if use_real else ":green_circle: Mock Generator"
    st.markdown(f"### :satellite_antenna: Data Source\n**Active:** {src}")
    if use_real:
        st.warning("Ensure the LoRa gateway is connected to COM3 at 9600 baud.")

    st.divider()
    if st.button(":counterclockwise_arrows_button: Reset Dashboard", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────
# Live Dashboard (auto-refreshing fragment)
# ─────────────────────────────────────────────
@st.fragment(run_every=timedelta(seconds=1))
def live_dashboard():
    """
    Runs every 1 s.  Generates readings, updates sliding windows,
    performs Z-score anomaly detection, and renders the full UI.
    """
    perf_start = time.perf_counter()
    st.session_state.tick += 1
    use_real = st.session_state.get("use_real_serial", False)
    now = datetime.now()
    ts  = now.strftime("%H:%M:%S")

    # ── Ingest data ──────────────────────────
    if use_real:
        # Real serial — one message per tick, node determined by gateway
        raw = get_serial_data(use_real=True)
        raws = [(raw, None)] if raw else []
    else:
        # Mock — one reading per node per tick for a fuller display
        raws = [(get_mock_serial_data(node), node) for node in NODES]

    for raw, _ in raws:
        if raw is None:
            continue

        reading = parse_serial_data(raw)
        if reading is None:
            continue

        node_id = reading["node"]

        # Sliding windows + anomaly check
        anomalies = {}
        for sensor in SENSORS:
            val = reading[sensor]
            window = st.session_state.windows[node_id][sensor]
            window.append(val)
            _, _, _, is_anom = check_anomaly(window, val)
            anomalies[sensor] = is_anom

            # Chart time-series (capped at MAX_CHART_POINTS)
            chart = st.session_state.chart_data[node_id][sensor]
            chart["times"].append(ts)
            chart["values"].append(val)
            if len(chart["times"]) > MAX_CHART_POINTS:
                chart["times"]  = chart["times"][-MAX_CHART_POINTS:]
                chart["values"] = chart["values"][-MAX_CHART_POINTS:]

        if any(anomalies.values()):
            st.session_state.anomaly_count += 1

        final_risk = correlate_risk(reading["hw_risk"], anomalies)
        st.session_state.node_risk[node_id]      = final_risk
        st.session_state.node_anomalies[node_id]  = anomalies
        st.session_state.latest_readings[node_id] = {s: reading[s] for s in SENSORS}

        st.session_state.serial_log.append(f"[{ts}] {raw}")

    proc_ms = (time.perf_counter() - perf_start) * 1000

    # ── RENDER ───────────────────────────────

    # Header
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <h1>MINEGUARD AI</h1>
            <p class="subtitle">Real-Time Mine Safety Monitoring &amp; Anomaly Detection</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Metric Cards ─────────────────────
    m1, m2, m3, m4 = st.columns(4)

    # System status = worst across all nodes
    risks = list(st.session_state.node_risk.values())
    if "CRITICAL" in risks:
        sys_status, sys_cls = "CRITICAL", "critical"
    elif "WARNING" in risks:
        sys_status, sys_cls = "WARNING", "warning"
    else:
        sys_status, sys_cls = "SAFE", "safe"

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">System Status</div>'
            f'<div class="metric-value {sys_cls}">{sys_status}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Active Nodes</div>'
            f'<div class="metric-value cyan">{len(NODES)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m3:
        lat_cls = "green" if proc_ms < 10 else "warning"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Pipeline Latency</div>'
            f'<div class="metric-value {lat_cls}">{proc_ms:.1f} ms</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with m4:
        anom_cls = "critical" if st.session_state.anomaly_count > 0 else "cyan"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Anomalies Detected</div>'
            f'<div class="metric-value {anom_cls}">{st.session_state.anomaly_count}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Node Monitoring Section ──────────────
    st.markdown(
        '<div class="section-header">Node Monitoring</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    risk_color_map = {
        "SAFE": "#00e676", "WARNING": "#ffab00", "CRITICAL": "#ff1744",
    }

    for idx, node in enumerate(NODES):
        with cols[idx]:
            risk       = st.session_state.node_risk[node]
            anoms      = st.session_state.node_anomalies[node]
            latest     = st.session_state.latest_readings[node]
            risk_lower = risk.lower()

            # Build sensor rows HTML
            rows_html = ""
            for sensor in SENSORS:
                val     = latest[sensor]
                is_anom = anoms.get(sensor, False)
                cls     = "anomaly" if is_anom else "normal"
                badge   = (
                    '<span class="anomaly-badge">ANOMALY</span>'
                    if is_anom
                    else '<span class="ok-badge">OK</span>'
                )
                rows_html += (
                    f'<div class="sensor-row">'
                    f'<span class="sensor-name">{sensor}</span>'
                    f'<span class="sensor-val {cls}">'
                    f'{val:.2f} {SENSOR_UNITS[sensor]}</span>'
                    f'{badge}'
                    f'</div>'
                )

            rc = risk_color_map.get(risk, "#8b949e")
            st.markdown(
                f'<div class="node-card {risk_lower}">'
                f'<div class="node-title">'
                f'<span class="status-dot {risk_lower}"></span> {node}'
                f'<span style="margin-left:auto;font-size:0.7rem;'
                f'color:{rc};font-weight:700;">{risk}</span>'
                f'</div>'
                f'{rows_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Plotly chart (only when there is data)
            has_data = any(
                len(st.session_state.chart_data[node][s]["values"]) > 0
                for s in SENSORS
            )
            if has_data:
                fig = build_node_chart(node)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"chart_{node}_{st.session_state.tick}",
                )

    # ── Serial Gateway Log ───────────────────
    st.markdown(
        '<div class="section-header">Serial Gateway Log</div>',
        unsafe_allow_html=True,
    )

    log_lines = list(st.session_state.serial_log)
    log_html  = "".join(
        f'<div class="serial-line">'
        f'<span class="ts">{line[:10]}</span> '
        f'<span class="arrow">&#9656;</span> '
        f'{line[11:]}'
        f'</div>'
        for line in reversed(log_lines)
    )
    st.markdown(
        f'<div class="serial-log">{log_html}</div>',
        unsafe_allow_html=True,
    )


# ── Launch ───────────────────────────────────
live_dashboard()
