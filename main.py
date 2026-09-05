import threading
import time

import dearpygui.dearpygui as dpg
import numpy as np
import soundcard as sc

WIDTH = 1000
HEIGHT = 710

SAMPLE_RATE = 48000
SAMPLE_COUNT = 2048

ROW_COUNT = SAMPLE_COUNT // 2 + 1
COL_COUNT = 256

BIN_WIDTH = SAMPLE_RATE / SAMPLE_COUNT

MULTIPLIER = 1.0
LOG_FILTER = True

is_recording = True

history_data = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.float32)

heatmap_data = np.zeros((ROW_COUNT, COL_COUNT, 4), dtype=np.float32)
heatmap_data[:, :, 3] = 1.0

heatmap_dirty = False

capture_data = np.zeros(ROW_COUNT, dtype=np.float32)
capture_flag = 0
capture_stop = threading.Event()
capture_thread = None


def capture_routine(device, sample_rate, sample_count):
    global capture_data
    global capture_flag

    HANN_WINDOW = np.hanning(sample_count)

    with device.recorder(sample_rate, 1, 512) as stream:
        while dpg.is_dearpygui_running and not capture_stop.is_set():
            if is_recording:
                data = stream.record(sample_count).flatten() * HANN_WINDOW

                fft_data = np.abs(np.fft.rfft(data))

                capture_data = fft_data
                capture_flag += 1
            else:
                time.sleep(0.1)

    capture_stop.clear()


def start_capture(device, sample_rate, sample_count):
    global capture_thread

    if capture_thread is not None and capture_thread.is_alive():
        capture_stop.set()
        capture_thread.join()

    capture_thread = threading.Thread(target=capture_routine, daemon=True, args=(device, sample_rate, sample_count))
    capture_thread.start()


def rebuild_heatmap():
    history = history_data.copy()

    if LOG_FILTER:
        history = np.clip((np.log10(history + 1e-6) / 3), 0.0, 1.0)

    history *= MULTIPLIER

    heatmap_data[:, :, 0] = history * 0.1
    heatmap_data[:, :, 1] = history * 0.5
    heatmap_data[:, :, 2] = history * 1.0

    dpg.set_value("heatmap_texture", heatmap_data)


def slider_callback(sender, value):
    global heatmap_dirty
    global MULTIPLIER

    if sender == "heatmap_crop_slider":
        dpg.set_axis_limits("spectrogram_y_axis", 0, value)
    elif sender == "amplifier_slider":
        MULTIPLIER = value
        heatmap_dirty = True


def checkbox_callback(sender, value):
    global heatmap_dirty
    global is_recording
    global LOG_FILTER

    if sender == "log_filter_checkbox":
        LOG_FILTER = value
        heatmap_dirty = True
    if sender == "recording_checkbox":
        is_recording = value


def combo_callback(sender, value):
    if sender == "source_combo":
        if value == "Default Microphone":
            start_capture(
                sc.default_microphone(),
                SAMPLE_RATE,
                SAMPLE_COUNT,
            )
        elif value == "System Audio":
            start_capture(
                sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True),
                SAMPLE_RATE,
                SAMPLE_COUNT,
            )


start_capture(sc.default_microphone(), SAMPLE_RATE, SAMPLE_COUNT)

dpg.create_context()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.texture_registry():
    dpg.add_dynamic_texture(COL_COUNT, ROW_COUNT, heatmap_data, tag="heatmap_texture")

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True, no_collapse=True):
    dpg.add_combo(
        label="Source",
        items=["Default Microphone", "System Audio"],
        default_value="Default Microphone",
        callback=combo_callback,
        tag="source_combo",
    )

    dpg.add_checkbox(
        label="Enable Recording",
        default_value=True,
        callback=checkbox_callback,
        tag="recording_checkbox",
    )

with dpg.window(label="Spectrogram", pos=(260, 10), width=731, height=480, no_close=True, no_scrollbar=True):
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
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (Samples)", tag="spectrogram_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Frequency (Hz)", tag="spectrogram_y_axis")

        dpg.add_image_series(
            texture_tag="heatmap_texture",
            bounds_min=(0, 0),
            bounds_max=(COL_COUNT, SAMPLE_RATE / 2 + 1),
            uv_min=(0.0, 1.0),
            uv_max=(1.0, 0.0),
            parent="spectrogram_x_axis",
        )

        dpg.set_axis_limits_constraints("spectrogram_x_axis", 0.0, COL_COUNT)
        dpg.set_axis_limits_constraints("spectrogram_y_axis", 0.0, SAMPLE_RATE / 2 + 1)

with dpg.window(label="Spectrum Analyzer", pos=(260, 500), width=731, height=200, no_close=True):  # noqa: SIM117
    with dpg.plot(width=-1, height=-1, tag="spectrum_plot"):
        dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="spectrum_x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="spectrum_y_axis", lock_min=True)

        dpg.add_bar_series(
            x=np.arange(ROW_COUNT),
            y=capture_data,
            parent="spectrum_x_axis",
            tag="spectrum_series",
        )

        dpg.set_axis_limits("spectrum_x_axis", 0.0, SAMPLE_RATE / 2)
        dpg.set_axis_limits_constraints("spectrum_y_axis", 0.0, 10.0)

dpg.setup_dearpygui()
dpg.show_viewport()

last_capture = 0

row = np.zeros(COL_COUNT, dtype=np.float32)

while dpg.is_dearpygui_running():
    if capture_flag > last_capture:
        row = capture_data.copy()
        last_capture = capture_flag

        history_data[:, 1:] = history_data[:, :-1]
        history_data[:, 0] = row

        if LOG_FILTER:
            row = np.clip((np.log10(row + 1e-6) / 3), 0.0, 1.0)

        row *= MULTIPLIER

        heatmap_data[:, 1:, :3] = heatmap_data[:, :-1, :3]
        heatmap_data[:, 0, 0] = row * 0.1
        heatmap_data[:, 0, 1] = row * 0.5
        heatmap_data[:, 0, 2] = row * 1.0

        dpg.set_value("heatmap_texture", heatmap_data)

        dpg.set_value("spectrum_series", (np.arange(SAMPLE_COUNT / 2) * BIN_WIDTH, row))

    if heatmap_dirty:
        rebuild_heatmap()

    dpg.render_dearpygui_frame()

dpg.destroy_context()

if capture_thread is not None and capture_thread.is_alive():
    capture_stop.set()
    capture_thread.join()
