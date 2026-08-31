import threading

import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 600

SAMPLE_RATE = 48000 * 2
SAMPLES = 2048

ROW_COUNT = 256
COL_COUNT = SAMPLES // 2 + 1

capture_data = np.zeros(COL_COUNT, dtype=np.float32)
capture_flag = 0


def capture_routine():
    global capture_data
    global capture_flag

    HANN_WINDOW = np.hanning(SAMPLES)

    device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    with device.recorder(SAMPLE_RATE, 1, SAMPLES) as stream:
        while dpg.is_dearpygui_running:
            data = stream.record(SAMPLES).flatten() * HANN_WINDOW

            fft_data = np.abs(np.fft.fft(data)[:COL_COUNT])
            fft_data = np.clip(np.log10(fft_data + 1e-6) / 2.0, 0.0, 1.0)

            capture_data = fft_data
            capture_flag += 1


capture_thread = threading.Thread(target=capture_routine, daemon=True)
capture_thread.start()


def crop_heatmap(sender, value):
    dpg.set_axis_limits("x_axis", 0, value)


history_data = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.float32)
heatmap_data = np.zeros((ROW_COUNT, COL_COUNT, 4), dtype=np.float32)

dpg.create_context()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(COL_COUNT, ROW_COUNT, heatmap_data, tag="heatmap_texture")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True):
    dpg.add_slider_float(
        label="Max Hz",
        default_value=(SAMPLE_RATE / 2),
        min_value=100.0,
        max_value=(SAMPLE_RATE / 2),
        callback=crop_heatmap,
        tag="heatmap_crop_slider",
    )

with dpg.window(label="Spectrogram", pos=(250, 0), width=750, height=600, no_close=True):  # noqa: SIM117
    with dpg.plot(width=-1, height=-1, tag="spectrogram_plot"):
        dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Time (Frames)", tag="y_axis")

        dpg.add_image_series(
            texture_tag="heatmap_texture", bounds_min=(0, 0), bounds_max=(SAMPLE_RATE / 2, ROW_COUNT), parent="x_axis"
        )

dpg.setup_dearpygui()
dpg.show_viewport()

last_capture = 0

while dpg.is_dearpygui_running():
    if capture_flag > last_capture:
        row = capture_data
        last_capture = capture_flag

        history_data[1:, :] = history_data[:-1, :]
        history_data[0, :] = row

        heatmap_colors = np.flipud(heatmap_data.reshape(ROW_COUNT, COL_COUNT, 4))
        heatmap_colors[:, :, 0] = 0.0
        heatmap_colors[:, :, 1] = history_data
        heatmap_colors[:, :, 2] = 0.0
        heatmap_colors[:, :, 3] = 1.0

        dpg.set_value("heatmap_texture", heatmap_data)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
