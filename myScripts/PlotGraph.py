import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import tkinter as tk
import pandas
import pickle

pattern = list()

try:
    with open('bufferFiles/BANKBARODA-EQ-restoreChartValues.pkl', "rb") as f:
        df = pickle.load(f)
        pattern.extend(df['direction'].tolist())
except:
    pass

# Get screen resolution
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

for i in range(0,len(pattern)):
    if pattern[i] ==0:
        pattern[i] = -1

# Build bricks list (x, y)
x, y = 0, 0
bricks = [(x, y)]
for move in pattern:
    x += 1
    y += move
    bricks.append((x, y))

# DPI and figure dimensions
dpi = 100
window_size = 200
fig_width = screen_width / dpi
fig_height = screen_height / dpi

# Create figure
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
ax.set_position([0, 0.12, 1, 0.88])  # Use full width and most of height
ax.set_aspect('equal')
ax.axis('off')

def draw_window(start):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(window_size):
        index = start + i
        if index >= len(bricks) - 1:
            break
        _, y0 = bricks[index]
        _, y1 = bricks[index + 1]
        color = 'green' if pattern[index] == 1 else 'red'
        brick_width = 1 # increase horizontal size
        brick_height = 1  # increase vertical size
        rect = patches.Rectangle((i * brick_width, y1 * brick_height),
                                 brick_width, brick_height,
                                 edgecolor='black', facecolor=color)

        ax.add_patch(rect)

    # Adjust x-limits so first and last bricks touch window edges
    ax.set_xlim(-0.5, window_size - 0.5)
    ax.set_ylim(min(y for _, y in bricks) - 2, max(y for _, y in bricks) + 2)
    plt.draw()

draw_window(0)

# Slider
ax_slider = plt.axes([0.15, 0.04, 0.7, 0.03])
slider = Slider(ax_slider, 'Scroll', 0, len(bricks) - window_size - 1, valinit=0, valstep=1)

def update(val):
    draw_window(int(slider.val))

slider.on_changed(update)

plt.title("Scrollable Renko Chart — Edge to Edge", pad=20)
plt.show()
