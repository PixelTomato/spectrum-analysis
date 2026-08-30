import numpy as np
import pygame as pg
import soundcard as sc

pg.init()

WIDTH, HEIGHT = 1044, 512

FREQUENCY = 48000
SAMPLES = 2048

HANN_WINDOW = np.hanning(SAMPLES)

screen = pg.display.set_mode((WIDTH, HEIGHT), pg.RESIZABLE, vsync=1)
pg.display.set_caption("Spectrum Analysis")

history = np.ndarray((SAMPLES // 2 + 1, 1024))

clock = pg.time.Clock()

device = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)


def draw_heatmap(x, y, width, height):
    heatmap = pg.Surface((SAMPLES // 2, height))

    pixels = pg.surfarray.pixels3d(heatmap)
    pixels[:, :, 1] = history[: SAMPLES // 2, :height]
    del pixels

    screen.blit(pg.transform.smoothscale(heatmap, (width, height)), (x, y))

    pg.draw.rect(screen, (120, 120, 120), (x, y, width, height), 2)


def draw_visualizer(x, y, width, height):
    bars = pg.Surface((SAMPLES // 2, 100))

    for i, amp in enumerate(history[:, 0] / 4):
        pg.draw.rect(bars, (0, 255, 0), (i, y + 100 - amp, 1, amp))

    screen.blit(pg.transform.smoothscale(bars, (width, height)), (x, y))

    pg.draw.rect(screen, (120, 120, 120), (x, y, width, height), 2)


with device.recorder(FREQUENCY, 1, SAMPLES) as stream:
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h

        data = stream.record(SAMPLES).flatten() * HANN_WINDOW

        amp = np.abs(np.fft.fft(data)[: SAMPLES // 2 + 1])
        amp = 20 * np.log10(amp + 0.000001)

        row = 2 * amp / SAMPLES * 7500
        row[0] /= 2
        row[-1] /= 2
        row = np.clip(row, 0, 255)

        history[:, 1:] = history[:, :-1]
        history[:, 0] = row

        screen.fill((0, 0, 0))

        draw_visualizer(10, 10, WIDTH - 20, 70)
        draw_heatmap(10, 90, WIDTH - 20, HEIGHT - 100)

        pg.display.flip()

        clock.tick()

pg.quit()
