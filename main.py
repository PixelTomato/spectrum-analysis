import numpy
import pyaudio
import pygame

pygame.init()

WIDTH, HEIGHT = 1044, 512

FREQUENCY = 44100 * 2
SAMPLES = 2048

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE, vsync=1)
pygame.display.set_caption("Spectrum Analysis")

history = numpy.ndarray((SAMPLES // 2 + 1, 1024))

clock = pygame.time.Clock()

audio = pyaudio.PyAudio()

stream = audio.open(FREQUENCY, 1, pyaudio.paInt16, True, frames_per_buffer=SAMPLES)


def draw_heatmap(x, y, width, height):
    heatmap = pygame.Surface((SAMPLES // 2, height))

    pixels = pygame.surfarray.pixels3d(heatmap)
    pixels[:, :, 1] = history[: SAMPLES // 2, :height]
    del pixels

    screen.blit(pygame.transform.smoothscale(heatmap, (width, height)), (x, y))

    pygame.draw.rect(screen, (120, 120, 120), (x, y, width, height), 2)


def draw_visualizer(x, y, width, height):
    BAR_WIDTH = numpy.ceil(width / (SAMPLES // 2))
    for i, amp in enumerate(history[:, 0] / 4):
        pygame.draw.rect(
            screen, (0, 255, 0), (x + i * BAR_WIDTH, y + height - amp, BAR_WIDTH, amp)
        )

    pygame.draw.rect(screen, (120, 120, 120), (x, y, width, height), 2)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            WIDTH, HEIGHT = event.w, event.h

    data = numpy.frombuffer(stream.read(SAMPLES), numpy.int16)
    amp = numpy.abs(numpy.fft.fft(data)[: SAMPLES // 2 + 1])

    row = 2 * amp / SAMPLES
    row[0] /= 2
    row[-1] /= 2
    row = numpy.clip(row, 0, 255)

    history[:, 1:] = history[:, :-1]
    history[:, 0] = row

    screen.fill((0, 0, 0))

    draw_visualizer(10, 10, WIDTH - 20, 70)
    draw_heatmap(10, 90, WIDTH - 20, HEIGHT - 100)

    pygame.display.flip()

    clock.tick()

stream.stop_stream()
stream.close()

audio.terminate()

pygame.quit()
