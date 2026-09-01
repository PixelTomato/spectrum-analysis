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

history_data = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.float32)

heatmap_data = np.zeros((ROW_COUNT, COL_COUNT, 4), dtype=np.float32)
heatmap_data[:, :, 3] = 1.0

heatmap_dirty = False

capture_data = np.zeros(COL_COUNT, dtype=np.float32)
capture_flag = 0
capture_stop = threading.Event()
capture_thread = None


def capture_routine(sample_rate, sample_count):
    global capture_data
    global capture_flag

    HANN_WINDOW = np.hanning(sample_count)

    # device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

    device = sc.default_microphone()

    with device.recorder(sample_rate, 1, 512) as stream:
        while dpg.is_dearpygui_running and not capture_stop.is_set():
            data = stream.record(sample_count).flatten() * HANN_WINDOW

            fft_data = np.abs(np.fft.rfft(data)[:COL_COUNT])

            capture_data = fft_data
            capture_flag += 1

    capture_stop.clear()


def start_capture():
    global capture_thread

    if capture_thread is not None and capture_thread.is_alive():
        capture_stop.set()
        capture_thread.join()

    capture_thread = threading.Thread(target=capture_routine, daemon=True, args=(SAMPLE_RATE, SAMPLES))
    capture_thread.start()


def rebuild_heatmap():
    history = np.flipud(history_data.copy())

    if LOG_FILTER:
        history = np.clip((np.log10(history + 1e-6) / 3), 0.0, 1.0)

    history *= MULTIPLIER

    heatmap_data[:, :, 0] = history * 0.1
    heatmap_data[:, :, 1] = history * 0.5
    heatmap_data[:, :, 2] = history * 1.0

    dpg.set_value("heatmap_texture", heatmap_data.ravel())


def slider_callback(sender, value):
    global heatmap_dirty
    global MULTIPLIER

    if sender == "heatmap_crop_slider":
        dpg.set_axis_limits("x_axis", 0, value)
    elif sender == "amplifier_slider":
        MULTIPLIER = value
        heatmap_dirty = True


def checkbox_callback(sender, value):
    global heatmap_dirty
    global LOG_FILTER

    if sender == "log_filter_checkbox":
        LOG_FILTER = value
        heatmap_dirty = True


start_capture()

dpg.create_context()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(COL_COUNT, ROW_COUNT, heatmap_data, tag="heatmap_texture")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True, no_collapse=True):
    dpg.add_button(
        label="Restart Capture",
        callback=start_capture,
    )

with dpg.window(label="Spectrogram", pos=(260, 10), width=720, height=480, no_close=True, no_scrollbar=True):
    with dpg.child_window(border=True, width=200, height=-1):
        dpg.add_slider_float(
            label="Max Hz",
            default_value=(SAMPLE_RATE / 2),
            min_value=100.0,
            max_value=(SAMPLE_RATE / 2),
            callback=slider_callback,
            width=-50,
            tag="heatmap_crop_slider",
        )

        dpg.add_slider_float(
            label="Amp",
            default_value=1.0,
            min_value=0.1,
            max_value=10.0,
            callback=slider_callback,
            width=-50,
            tag="amplifier_slider",
        )

        dpg.add_checkbox(
            label="Log Filter",
            default_value=True,
            callback=checkbox_callback,
            tag="log_filter_checkbox",
        )

    with dpg.plot(pos=(216, 27), width=-1, height=-1, tag="spectrogram_plot"):
        dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Time (Frames)", tag="y_axis")

        dpg.add_image_series(
            texture_tag="heatmap_texture",
            bounds_min=(0, 0),
            bounds_max=(SAMPLE_RATE / 2, ROW_COUNT),
            parent="x_axis",
        )

dpg.setup_dearpygui()
dpg.show_viewport()

last_capture = 0

row = np.zeros(COL_COUNT, dtype=np.float32)

while dpg.is_dearpygui_running():
    if capture_flag > last_capture:
        row = capture_data.copy()
        last_capture = capture_flag

        history_data[1:, :] = history_data[:-1, :]
        history_data[0, :] = row

        if LOG_FILTER:
            row = np.clip((np.log10(row + 1e-6) / 3), 0.0, 1.0)

        row *= MULTIPLIER

        heatmap_data[:-1, :, :3] = heatmap_data[1:, :, :3]
        heatmap_data[-1, :, 0] = row * 0.1
        heatmap_data[-1, :, 1] = row * 0.5
        heatmap_data[-1, :, 2] = row * 1.0

    if heatmap_dirty:
        rebuild_heatmap()
    else:
        dpg.set_value("heatmap_texture", heatmap_data.ravel())

    dpg.render_dearpygui_frame()

dpg.destroy_context()
