import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 600

FREQUENCY = 48000
SAMPLES = 2048

HANN_WINDOW = np.hanning(SAMPLES)

ROW_COUNT = 256
COL_COUNT = SAMPLES // 2 + 1

last_input = np.zeros(COL_COUNT, dtype=np.float32)

heatmap_texture = np.zeros(COL_COUNT * ROW_COUNT * 4, dtype=np.float32)

dpg.create_context()
dpg.show_metrics()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(COL_COUNT, ROW_COUNT, heatmap_texture, tag="heatmap")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True):
    dpg.add_text("Testing...")

with dpg.window(label="Spectrogram", pos=(250, 0), width=750, height=600, no_close=True, no_resize=True):
    dpg.add_image("heatmap", width=734, height=565)

dpg.setup_dearpygui()
dpg.show_viewport()

device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

with device.recorder(FREQUENCY, 1, SAMPLES) as stream:
    while dpg.is_dearpygui_running():
        data = stream.record(SAMPLES).flatten() * HANN_WINDOW

        amp = np.abs(np.fft.fft(data)[: SAMPLES // 2 + 1])
        amp = np.log10(amp + 1e-6)

        history = heatmap_texture.reshape(ROW_COUNT, COL_COUNT, 4)
        history[1:] = history[:-1]
        history[0, :, 1] = amp
        history[0, :, 3] = 1.0
        dpg.set_value("heatmap", heatmap_texture)

        dpg.render_dearpygui_frame()

dpg.destroy_context()
