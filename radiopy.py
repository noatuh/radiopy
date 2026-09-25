#!/usr/bin/env python3
"""
RadioHTML Transceiver (Desktop Controller & Web Dashboard)
- Desktop GUI (Tkinter): Select audio input/output devices, view server logs, control server.
- Web GUI (Flask): Plain-text and Raw HTML transmission interface in the browser.
"""

import sys
import time
import zlib
import struct
import threading
import webbrowser
import html
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import sounddevice as sd
from flask import Flask, render_template_string, request, jsonify

# --- Modem Configuration ---
SAMPLE_RATE = 44100
BAUD_RATE = 1200
FREQ_MARK = 1200.0   # Logical 1
FREQ_SPACE = 2200.0  # Logical 0
SYNC_WORD = b"\xD3\x91"
PREAMBLE_BYTES = 16

# Shared Configuration & Thread-safe Logging Queue
device_config = {
    "input_device": None,   # Device ID integer
    "output_device": None   # Device ID integer
}

log_queue = queue.Queue()

def log(msg: str):
    """Thread-safe logging helper."""
    timestamp = time.strftime("[%H:%M:%S]")
    log_queue.put(f"{timestamp} {msg}\n")


# --- Core Modem Engine ---
class RadioHTMLModem:
    def __init__(self, sample_rate=SAMPLE_RATE, baud_rate=BAUD_RATE):
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.spb = int(sample_rate / baud_rate)

    def pack_html(self, html_string: str) -> bytes:
        payload = zlib.compress(html_string.encode('utf-8'))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = SYNC_WORD + struct.pack(">H", len(payload))
        footer = struct.pack(">I", crc)
        return (b"\x55" * PREAMBLE_BYTES) + header + payload + footer

    def modulate_afsk(self, data_bytes: bytes) -> np.ndarray:
        bits = []
        for byte in data_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        bits = np.array(bits, dtype=np.uint8)

        total_samples = len(bits) * self.spb
        freqs = np.where(bits == 1, FREQ_MARK, FREQ_SPACE)
        repeated_freqs = np.repeat(freqs, self.spb)
        phase = 2.0 * np.pi * np.cumsum(repeated_freqs) / self.sample_rate
        return (0.8 * np.sin(phase)).astype(np.float32)

    def transmit(self, html_content: str, device_id=None):
        frame = self.pack_html(html_content)
        audio = self.modulate_afsk(frame)
        time.sleep(0.3)  # Small PTT setup delay
        sd.play(audio, samplerate=self.sample_rate, device=device_id)
        sd.wait()

    def demodulate_afsk(self, audio_samples: np.ndarray) -> np.ndarray:
        num_bits = len(audio_samples) // self.spb
        demodulated_bits = []

        t_bit = np.arange(self.spb) / self.sample_rate
        mark_cos, mark_sin = np.cos(2*np.pi*FREQ_MARK*t_bit), np.sin(2*np.pi*FREQ_MARK*t_bit)
        space_cos, space_sin = np.cos(2*np.pi*FREQ_SPACE*t_bit), np.sin(2*np.pi*FREQ_SPACE*t_bit)

        for i in range(num_bits):
            chunk = audio_samples[i * self.spb : (i + 1) * self.spb]
            if len(chunk) < self.spb:
                break
            mark_p = np.sum(chunk * mark_cos)**2 + np.sum(chunk * mark_sin)**2
            space_p = np.sum(chunk * space_cos)**2 + np.sum(chunk * space_sin)**2
            demodulated_bits.append(1 if mark_p > space_p else 0)

        return np.array(demodulated_bits, dtype=np.uint8)

    def unpack_and_verify(self, bits: np.ndarray) -> str:
        byte_list = []
        for i in range(0, len(bits) - 7, 8):
            val = 0
            for b in range(8):
                val = (val << 1) | bits[i + b]
            byte_list.append(val)
        raw_bytes = bytes(byte_list)

        sync_idx = raw_bytes.find(SYNC_WORD)
        if sync_idx == -1:
            raise ValueError("No valid transmission signal detected.")

        payload_start = sync_idx + len(SYNC_WORD) + 2
        header = raw_bytes[sync_idx + len(SYNC_WORD) : payload_start]
        length = struct.unpack(">H", header)[0]
        payload_end = payload_start + length
        crc_end = payload_end + 4

        payload = raw_bytes[payload_start:payload_end]
        received_crc = struct.unpack(">I", raw_bytes[payload_end:crc_end])[0]

        if (zlib.crc32(payload) & 0xFFFFFFFF) != received_crc:
            raise ValueError("Checksum failed. Corrupted audio packet.")

        return zlib.decompress(payload).decode('utf-8')

    def receive(self, seconds=10, device_id=None) -> str:
        audio = sd.rec(int(seconds * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32', device=device_id)
        sd.wait()
        bits = self.demodulate_afsk(audio.flatten())
        return self.unpack_and_verify(bits)


# --- HTML Template Generator ---
THEMES = {
    "info": { "bg": "#0f172a", "border": "#38bdf8", "title_color": "#38bdf8", "icon": "ℹ️" },
    "alert": { "bg": "#450a0a", "border": "#ef4444", "title_color": "#fca5a5", "icon": "🚨" },
    "success": { "bg": "#064e3b", "border": "#10b981", "title_color": "#a7f3d0", "icon": "✅" },
    "warning": { "bg": "#451a03", "border": "#f59e0b", "title_color": "#fde68a", "icon": "⚠️" }
}

def generate_html_card(title: str, text: str, theme_key: str = "info", station: str = "Base Station") -> str:
    theme = THEMES.get(theme_key, THEMES["info"])
    clean_title = html.escape(title if title.strip() else "INCOMING TRANSMISSION")
    clean_station = html.escape(station if station.strip() else "Radio Transceiver")
    paragraphs = html.escape(text).split('\n')
    formatted_body = "".join([f"<p style='margin: 8px 0; line-height: 1.4;'>{p}</p>" for p in paragraphs if p.strip()])
    
    if not formatted_body:
        formatted_body = "<p style='margin: 8px 0;'><em>(Empty message body)</em></p>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #000; margin: 0; padding: 12px; }}
        .card {{ background: {theme['bg']}; border: 2px solid {theme['border']}; border-radius: 10px; padding: 16px; color: #f8fafc; max-width: 100%; box-sizing: border-box; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 8px; margin-bottom: 12px; }}
        .title {{ font-size: 1.1rem; font-weight: bold; color: {theme['title_color']}; margin: 0; }}
        .meta {{ font-size: 0.75rem; opacity: 0.7; font-family: monospace; }}
        .body {{ font-size: 0.95rem; color: #e2e8f0; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h2 class="title">{theme['icon']} {clean_title}</h2>
            <span class="meta">{clean_station}</span>
        </div>
        <div class="body">
            {formatted_body}
        </div>
    </div>
</body>
</html>"""


# --- Flask Web Server ---
app = Flask(__name__)
modem = RadioHTMLModem()

state = {
    "is_transmitting": False,
    "is_receiving": False,
    "last_status": "Idle. Ready to transmit or receive.",
    "received_html": "<em>No pages received yet. Press 'Listen & Receive' in the browser to start decoding.</em>"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RadioHTML Web Dashboard</title>
    <style>
        :root { --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; --muted: #94a3b8; }
        body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid #334155; }
        h1 { margin: 0; font-size: 1.4rem; color: var(--accent); }
        .status-badge { background: #334155; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .card { background: var(--card); padding: 20px; border-radius: 12px; border: 1px solid #334155; }
        h2 { margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        
        .mode-switcher { display: flex; gap: 6px; margin-bottom: 16px; background: #0f172a; padding: 4px; border-radius: 8px; border: 1px solid #334155; }
        .tab-btn { flex: 1; background: transparent; color: var(--muted); border: none; padding: 8px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; text-align: center; transition: 0.2s; }
        .tab-btn.active { background: var(--accent); color: #0f172a; }

        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
        input, select, textarea { width: 100%; background: #0f172a; color: var(--text); border: 1px solid #334155; border-radius: 6px; padding: 10px; font-size: 0.95rem; box-sizing: border-box; }
        textarea { height: 130px; resize: vertical; }
        textarea.code-mode { font-family: monospace; height: 260px; color: #38bdf8; }
        
        .btn-action { background: var(--accent); color: #0f172a; border: none; padding: 12px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; width: 100%; font-size: 1rem; }
        .btn-action:hover { opacity: 0.9; }
        .btn-action:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .viewport { width: 100%; height: 360px; background: #000; border-radius: 8px; border: 1px solid #334155; margin-top: 10px; }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📻 RadioHTML Web Dashboard</h1>
            <div id="status" class="status-badge">System Ready</div>
        </header>

        <div class="grid">
            <div class="card">
                <h2>📤 Transmit Panel</h2>
                
                <div class="mode-switcher">
                    <button id="tabText" class="tab-btn active" onclick="setMode('text')">📝 Plain Text Mode</button>
                    <button id="tabHtml" class="tab-btn" onclick="setMode('html')">💻 Raw HTML Mode</button>
                </div>

                <div id="textModeForm">
                    <div class="form-group">
                        <label>Station Name / Callsign</label>
                        <input type="text" id="station" value="Station 1" placeholder="e.g. Base Camp">
                    </div>

                    <div class="form-group">
                        <label>Message Title</label>
                        <input type="text" id="title" value="STATUS UPDATE" placeholder="e.g. Weather Alert">
                    </div>

                    <div class="form-group">
                        <label>Card Style / Theme</label>
                        <select id="theme">
                            <option value="info">ℹ️ Normal Info (Blue)</option>
                            <option value="alert">🚨 Emergency Alert (Red)</option>
                            <option value="success">✅ Success / Safe (Green)</option>
                            <option value="warning">⚠️ Warning / Notice (Yellow)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Message Text</label>
                        <textarea id="message" placeholder="Type your message here...">All systems operational. Radio check passed on Channel 4.</textarea>
                    </div>
                </div>

                <div id="htmlModeForm" style="display: none;">
                    <div class="form-group">
                        <label>Raw HTML Document Code</label>
                        <textarea id="rawHtml" class="code-mode" placeholder="<html>...</html>"><div style="font-family:sans-serif; background:#0f172a; color:#fff; padding:15px; border-radius:8px;">
  <h2 style="color:#38bdf8; margin:0;">CUSTOM HTML PAGE</h2>
  <p>Transmitted directly as raw code.</p>
</div></textarea>
                    </div>
                </div>

                <button id="txBtn" class="btn-action" onclick="sendMessage()">Broadcast Over Radio</button>
            </div>

            <div class="card">
                <h2>📥 Received Message Display</h2>
                <button id="rxBtn" class="btn-action" onclick="startReceiver()" style="margin-bottom: 12px;">Listen & Receive (10 Seconds)</button>
                <iframe id="radioViewport" class="viewport" srcdoc="<em>Waiting for incoming transmission...</em>"></iframe>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'text';

        function setMode(mode) {
            currentMode = mode;
            if (mode === 'text') {
                document.getElementById('textModeForm').style.display = 'block';
                document.getElementById('htmlModeForm').style.display = 'none';
                document.getElementById('tabText').classList.add('active');
                document.getElementById('tabHtml').classList.remove('active');
            } else {
                document.getElementById('textModeForm').style.display = 'none';
                document.getElementById('htmlModeForm').style.display = 'block';
                document.getElementById('tabText').classList.remove('active');
                document.getElementById('tabHtml').classList.add('active');
            }
        }

        async function updateStatus() {
            const res = await fetch('/api/status');
            const data = await res.json();
            document.getElementById('status').innerText = data.last_status;
            
            if (data.received_html) {
                document.getElementById('radioViewport').srcdoc = data.received_html;
            }

            document.getElementById('txBtn').disabled = data.is_transmitting || data.is_receiving;
            document.getElementById('rxBtn').disabled = data.is_transmitting || data.is_receiving;
        }

        async function sendMessage() {
            let payload = { mode: currentMode };
            if (currentMode === 'text') {
                payload.station = document.getElementById('station').value;
                payload.title = document.getElementById('title').value;
                payload.theme = document.getElementById('theme').value;
                payload.message = document.getElementById('message').value;
            } else {
                payload.html = document.getElementById('rawHtml').value;
            }

            await fetch('/api/transmit', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
        }

        async function startReceiver() {
            await fetch('/api/receive', { method: 'POST' });
        }

        setInterval(updateStatus, 1000);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def get_status():
    return jsonify(state)

@app.route("/api/transmit", methods=["POST"])
def api_transmit():
    if state["is_transmitting"] or state["is_receiving"]:
        return jsonify({"error": "Busy"}), 400

    data = request.json or {}
    mode = data.get("mode", "text")
    
    if mode == "html":
        generated_html = data.get("html", "")
    else:
        generated_html = generate_html_card(
            title=data.get("title", ""),
            text=data.get("message", ""),
            theme_key=data.get("theme", "info"),
            station=data.get("station", "Radio Station")
        )

    def run_tx():
        state["is_transmitting"] = True
        state["last_status"] = "Broadcasting audio tones over radio..."
        log(f"[TX] Starting transmission using output device ID: {device_config['output_device']}")
        try:
            modem.transmit(generated_html, device_id=device_config["output_device"])
            state["last_status"] = "Broadcast finished successfully!"
            log("[TX] Transmission finished.")
        except Exception as e:
            state["last_status"] = f"TX Error: {str(e)}"
            log(f"[TX Error] {str(e)}")
        finally:
            state["is_transmitting"] = False

    threading.Thread(target=run_tx).start()
    return jsonify({"status": "started"})

@app.route("/api/receive", methods=["POST"])
def api_receive():
    if state["is_transmitting"] or state["is_receiving"]:
        return jsonify({"error": "Busy"}), 400

    def run_rx():
        state["is_receiving"] = True
        state["last_status"] = "Listening on microphone for 10 seconds..."
        log(f"[RX] Listening for audio using input device ID: {device_config['input_device']}")
        try:
            result = modem.receive(seconds=10, device_id=device_config["input_device"])
            state["received_html"] = result
            state["last_status"] = "Message received and rendered!"
            log("[RX] Message decoded successfully.")
        except Exception as e:
            state["last_status"] = f"RX Warning: {str(e)}"
            log(f"[RX Warning] {str(e)}")
        finally:
            state["is_receiving"] = False

    threading.Thread(target=run_rx).start()
    return jsonify({"status": "started"})


# --- Desktop GUI Application (Tkinter) ---
class RadioDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RadioHTML Desktop Control Center")
        self.root.geometry("640x520")
        self.root.configure(bg="#0f172a")

        self.input_devices = []
        self.output_devices = []

        self.setup_ui()
        self.refresh_audio_devices()
        self.poll_log_queue()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#0f172a", foreground="#f8fafc")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground="#1e293b", background="#334155", foreground="#ffffff")
        style.configure("TButton", background="#38bdf8", foreground="#0f172a", font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", "#7dd3fc")])

        # Header Frame
        header_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=12)
        header_frame.pack(fill="x")

        tk.Label(
            header_frame, text="📻 RadioHTML Desktop Controller",
            font=("Segoe UI", 14, "bold"), fg="#38bdf8", bg="#1e293b"
        ).pack(side="left")

        btn_open_browser = ttk.Button(header_frame, text="🌐 Open Web GUI", command=self.open_browser)
        btn_open_browser.pack(side="right")

        # Device Selection Frame
        device_frame = tk.LabelFrame(
            self.root, text=" Audio Device Selection ",
            font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1e293b", padx=15, pady=12
        )
        device_frame.pack(fill="x", padx=15, pady=15)

        # Input Device (Microphone)
        tk.Label(device_frame, text="Input Device (Radio Speaker -> PC Mic):", bg="#1e293b", fg="#94a3b8").grid(row=0, column=0, sticky="w", pady=4)
        self.combo_input = ttk.Combobox(device_frame, state="readonly", width=50)
        self.combo_input.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.combo_input.bind("<<ComboboxSelected>>", self.on_input_selected)

        # Output Device (Speaker)
        tk.Label(device_frame, text="Output Device (PC Headphone -> Radio Mic):", bg="#1e293b", fg="#94a3b8").grid(row=2, column=0, sticky="w", pady=4)
        self.combo_output = ttk.Combobox(device_frame, state="readonly", width=50)
        self.combo_output.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.combo_output.bind("<<ComboboxSelected>>", self.on_output_selected)

        # Refresh Devices Button
        btn_refresh = tk.Button(
            device_frame, text="🔄 Refresh Audio Devices",
            command=self.refresh_audio_devices, bg="#334155", fg="#ffffff",
            activebackground="#475569", activeforeground="#ffffff", bd=0, padx=10, pady=5
        )
        btn_refresh.grid(row=4, column=0, sticky="w", pady=5)

        device_frame.columnconfigure(0, weight=1)

        # Log Frame
        log_frame = tk.LabelFrame(
            self.root, text=" Real-Time Server Console Logs ",
            font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1e293b", padx=15, pady=10
        )
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.log_text = scrolledtext.ScrolledText(
            log_frame, bg="#0f172a", fg="#38bdf8",
            insertbackground="white", font=("Consolas", 9), bd=0
        )
        self.log_text.pack(fill="both", expand=True)

    def refresh_audio_devices(self):
        """Queries sounddevice library and populates comboboxes."""
        devices = sd.query_devices()
        self.input_devices = []
        self.output_devices = []

        input_names = []
        output_names = []

        for idx, dev in enumerate(devices):
            name = f"[{idx}] {dev['name']}"
            if dev['max_input_channels'] > 0:
                self.input_devices.append((idx, dev['name']))
                input_names.append(name)
            if dev['max_output_channels'] > 0:
                self.output_devices.append((idx, dev['name']))
                output_names.append(name)

        self.combo_input['values'] = input_names
        self.combo_output['values'] = output_names

        # Select default device or first available
        if input_names:
            self.combo_input.current(0)
            self.on_input_selected(None)

        if output_names:
            self.combo_output.current(0)
            self.on_output_selected(None)

        log("Audio devices refreshed.")

    def on_input_selected(self, event):
        idx = self.combo_input.current()
        if 0 <= idx < len(self.input_devices):
            dev_id = self.input_devices[idx][0]
            device_config["input_device"] = dev_id
            log(f"Set Input Device to ID {dev_id}: {self.input_devices[idx][1]}")

    def on_output_selected(self, event):
        idx = self.combo_output.current()
        if 0 <= idx < len(self.output_devices):
            dev_id = self.output_devices[idx][0]
            device_config["output_device"] = dev_id
            log(f"Set Output Device to ID {dev_id}: {self.output_devices[idx][1]}")

    def open_browser(self):
        webbrowser.open("http://127.0.0.1:5000")

    def poll_log_queue(self):
        """Reads logs from queue and displays in Tkinter text box."""
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            self.log_text.insert(tk.END, msg)
            self.log_text.see(tk.END)
        self.root.after(200, self.poll_log_queue)


# --- Application Startup ---
def run_flask_server():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    # Start Flask Web Server in background daemon thread
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    log("Started background web server at http://127.0.0.1:5000")

    # Automatically open browser GUI after 1.2 seconds
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    # Launch Native PC Desktop App
    root = tk.Tk()
    app_gui = RadioDesktopApp(root)
    root.mainloop()