import threading

import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 600

SAMPLE_RATE = 48000
SAMPLES = 2048

ROW_COUNT = 256
COL_COUNT = SAMPLES // 2 + 1

MULTIPLIER = 1.0
LOG_FILTER = True

capture_data = np.zeros(COL_COUNT, dtype=np.float32)
capture_flag = 0


def capture_routine():
    global capture_data
    global capture_flag

    HANN_WINDOW = np.hanning(SAMPLES)

    # device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    device = sc.default_microphone()

    with device.recorder(SAMPLE_RATE, 1, 512) as stream:
        while dpg.is_dearpygui_running:
            data = stream.record(SAMPLES).flatten() * HANN_WINDOW

            fft_data = np.abs(np.fft.fft(data)[:COL_COUNT])

            capture_data = fft_data
            capture_flag += 1


capture_thread = threading.Thread(target=capture_routine, daemon=True)
capture_thread.start()


def slider_callback(sender, value):
    global MULTIPLIER

    if sender == "heatmap_crop_slider":
        dpg.set_axis_limits("x_axis", 0, value)
    elif sender == "amplifier_slider":
        MULTIPLIER = value


def checkbox_callback(sender, value):
    global LOG_FILTER

    if sender == "log_filter_checkbox":
        LOG_FILTER = not LOG_FILTER


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
        callback=slider_callback,
        tag="heatmap_crop_slider",
    )

    dpg.add_slider_float(
        label="Multiplier",
        default_value=1.0,
        min_value=0.1,
        max_value=10.0,
        callback=slider_callback,
        tag="amplifier_slider",
    )

    dpg.add_checkbox(
        label="Log Filter",
        default_value=True,
        callback=checkbox_callback,
        tag="log_filter_checkbox",
    )

with dpg.window(label="Spectrogram", pos=(250, 0), width=480, height=360, no_close=True):  # noqa: SIM117
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

        v = history_data.copy()

        if LOG_FILTER:
            v = np.clip(np.log10(v + 1e-6) / 3, 0.0, 1.0)

        v *= MULTIPLIER

        heatmap_colors = np.flipud(heatmap_data.reshape(ROW_COUNT, COL_COUNT, 4))
        heatmap_colors[:, :, 0] = np.clip((-0.42 * v ** 3 + 1.25 * v ** 2 + 0.17 * v + 0.02), 0.0, 1.0)
        heatmap_colors[:, :, 1] = np.clip((1.83 * v ** 3 - 1.27 * v ** 2 + 0.44 * v), 0.0, 1.0)
        heatmap_colors[:, :, 2] = np.clip((-2.85 * v ** 3 + 5.16 * v ** 2 - 1.63 * v + 0.3 - 0.25 * np.exp(-120 * v)), 0.0, 1.0)
        heatmap_colors[:, :, 3] = 1.0

        dpg.set_value("heatmap_texture", heatmap_data)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
