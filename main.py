import threading

import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 600

FREQUENCY = 48000
SAMPLES = 2048

ROW_COUNT = 256
COL_COUNT = SAMPLES // 2 + 1

last_capture = np.zeros(COL_COUNT, dtype=np.float32)


def capture_routine():
    global last_capture

    HANN_WINDOW = np.hanning(SAMPLES)

    device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    with device.recorder(FREQUENCY, 1, SAMPLES) as stream:
        while dpg.is_dearpygui_running:
            data = stream.record(SAMPLES).flatten() * HANN_WINDOW

            fft_data = np.abs(np.fft.fft(data)[:COL_COUNT])
            fft_data = np.clip(np.log10(fft_data + 1e-6) / 2.0, 0.0, 1.0)

            last_capture = fft_data


capture_thread = threading.Thread(target=capture_routine, daemon=True)
capture_thread.start()

history_data = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.float32)
heatmap_data = np.zeros((ROW_COUNT, COL_COUNT, 4), dtype=np.float32)

dpg.create_context()
dpg.show_metrics()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(COL_COUNT, ROW_COUNT, heatmap_data, tag="heatmap_texture")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True):
    dpg.add_text("Testing...")

with dpg.window(label="Spectrogram", pos=(250, 0), width=750, height=600, no_close=True, no_resize=True):  # noqa: SIM117
    with dpg.plot(width=-1, height=-1, tag="spectrogram_plot"):
        dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Time (Frames)", tag="y_axis")

        dpg.add_image_series(
            texture_tag="heatmap_texture",
            bounds_min=(0, 0),
            bounds_max=(COL_COUNT, ROW_COUNT),
            parent="x_axis",
        )

dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    row = last_capture

    # history_data[1:, :] = history_data[:-1, :]
    history_data[1:, :] = history_data[:-1, :]
    history_data[0, :] = row

    heatmap_colors = heatmap_data.reshape(ROW_COUNT, COL_COUNT, 4)
    heatmap_colors[:, :, 0] = 0.0
    heatmap_colors[:, :, 1] = history_data
    heatmap_colors[:, :, 2] = 0.0
    heatmap_colors[:, :, 3] = 1.0

    dpg.set_value("heatmap_texture", heatmap_data)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
