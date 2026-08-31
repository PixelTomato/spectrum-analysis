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
            fft_data = np.clip(np.log10(fft_data + 1e-6) / 2.0, 0.0, 1.0)

            last_capture = fft_data


capture_thread = threading.Thread(target=capture_routine, daemon=True)
capture_thread.start()

heatmap_data = np.zeros(BIN_COUNT * ROW_COUNT, dtype=np.float32)

dpg.create_context()
dpg.show_metrics()
dpg.create_viewport(title="Spectrum Analysis", width=WIDTH, height=HEIGHT, vsync=True)

with dpg.window(label="Options", width=250, height=HEIGHT, no_close=True, no_move=True):
    dpg.add_text("Testing...")

with dpg.window(label="Spectrogram", pos=(250, 0), width=750, height=600, no_close=True, no_resize=True):
    with dpg.plot(width=-1, height=-1, tag="spectrogram_plot"):
        dpg.add_plot_axis(dpg.mvXAxis, label="Hz", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis")

        dpg.set_axis_limits("x_axis", 0, ROW_COUNT)
        dpg.set_axis_limits("y_axis", 0, BIN_COUNT)

        dpg.add_heat_series(
            x=heatmap_data,
            rows=ROW_COUNT,
            cols=BIN_COUNT,
            label="Audio",
            bounds_min=(0, 0),
            bounds_max=(ROW_COUNT, BIN_COUNT),
            format="",
            tag="spectrogram_series",
            parent="x_axis",
        )

        dpg.bind_colormap("spectrogram_plot", dpg.mvPlotColormap_Hot)


dpg.setup_dearpygui()
dpg.show_viewport()


while dpg.is_dearpygui_running():
    row = last_capture

    history = heatmap_data.reshape(ROW_COUNT, BIN_COUNT)
    history[1:, :] = history[:-1, :]
    history[0, :] = last_capture

    dpg.set_value("spectrogram_series", [heatmap_data])

    dpg.render_dearpygui_frame()

dpg.destroy_context()
