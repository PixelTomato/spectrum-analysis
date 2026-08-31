import threading

import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 600

FREQUENCY = 48000
SAMPLES = 2048

ROW_COUNT = 256
BIN_COUNT = SAMPLES // 2 + 1

last_capture = np.zeros(BIN_COUNT, dtype=np.float32)


def capture_routine():
    global last_capture

    HANN_WINDOW = np.hanning(SAMPLES)

    device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    with device.recorder(FREQUENCY, 1, SAMPLES) as stream:
        while dpg.is_dearpygui_running:
            data = stream.record(SAMPLES).flatten() * HANN_WINDOW

            fft_data = np.abs(np.fft.fft(data)[:BIN_COUNT])
            fft_data = np.clip(np.log10(fft_data + 1e-6), 0.0, 1.0)

            last_capture = fft_data


capture_thread = threading.Thread(target=capture_routine, daemon=True)
capture_thread.start()

heatmap_texture = np.zeros(BIN_COUNT * ROW_COUNT * 4, dtype=np.float32)

dpg.create_context()
dpg.show_metrics()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(BIN_COUNT, ROW_COUNT, heatmap_texture, tag="heatmap")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True):
    dpg.add_text("Testing...")

with dpg.window(label="Spectrogram", pos=(250, 0), width=750, height=600, no_close=True, no_resize=True):
    dpg.add_image("heatmap", width=734, height=565)

dpg.setup_dearpygui()
dpg.show_viewport()


while dpg.is_dearpygui_running():
    row = last_capture

    history = heatmap_texture.reshape(ROW_COUNT, BIN_COUNT, 4)
    history[1:] = history[:-1]
    history[0, :, 1] = row * 255.0
    history[0, :, 3] = 1.0
    dpg.set_value("heatmap", heatmap_texture)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
