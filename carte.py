import tkinter as tk
from functools import partial
import random
import time
import threading




def init_map(num):
    return [[0 for _ in range(num)] for _ in range(num)]

map = init_map(50)

colors = {
    0: "#CCFFCC", 
    1: "#FF0000", 
    2: "#006400"  
}

clicked_x = clicked_y = None

def cycle_trash():
    global x
    if x == 0:
        x = 1
        interval = 0.1  # Change this to set the interval in seconds
        def place_trash_periodically():
            while x == 1:
                place_random_trash()
                time.sleep(interval)
            x = 0
        threading.Thread(target=place_trash_periodically).start()


def select_block(start_x, start_y, end_x, end_y, new_color):
    for y in range(start_y, end_y+1):
        for x in range(start_x, end_x+1):
            map[y][x] = new_color

def draw_pixel(canvas, x, y, color, pixel_size, tag):
    canvas.create_rectangle(x*pixel_size, y*pixel_size, (x+1)*pixel_size, (y+1)*pixel_size, fill=color, outline='black', tags=tag)

def draw_map(canvas, window_size):
    pixel_size = 3*window_size // len(map)
    for y, row in enumerate(map):
        for x, value in enumerate(row):
            color = colors[value]
            tag = f"pixel{x}-{y}"
            draw_pixel(canvas, x, y, color, pixel_size, tag)
            canvas.tag_bind(tag, '<Button-1>', lambda event, x=x, y=y: on_click(event, x, y))
def creation_bloc(x, y):
    global clicked_x, clicked_y
    if clicked_x is None:
        clicked_x = x
        clicked_y = y
    else:
        start_x = min(clicked_x, x)
        start_y = min(clicked_y, y)
        end_x = max(clicked_x, x)
        end_y = max(clicked_y, y)
        
        # Create a menu
        menu = tk.Menu(root, tearoff=0)
    
        menu.add_command(label="Changer en obstacle", command=lambda: select_and_draw(start_x, start_y, end_x, end_y, 1))
        menu.add_command(label="Réinitialiser", command=lambda: select_and_draw(start_x, start_y, end_x, end_y, 0))
        menu.post(root.winfo_pointerx(), root.winfo_pointery())

        clicked_x = clicked_y = None

def select_and_draw(start_x, start_y, end_x, end_y, value):
    select_block(start_x, start_y, end_x, end_y, value)
    draw_map(canvas, window_size)
def on_click(event, x, y):
    # Create a menu
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Changer en obstacle", command=partial(change_to_red, x, y))
    menu.add_command(label="Changer en déchet", command=partial(change_to_green, x, y))
    menu.add_command(label="Réinitialiser", command=partial(change_to_white, x, y))
    menu.add_command(label="Faire un bloc", command=partial(creation_bloc, x, y))
    
    # Display the menu
    menu.post(event.x_root, event.y_root)

def change_to_red(x, y):
    map[y][x] = 1
    color = colors[1]
    tag = f"pixel{x}-{y}"
    draw_pixel(canvas, x, y, color, pixel_size, tag)
    print(map)
    
def change_to_white(x, y):
    map[y][x] = 0
    color = colors[0]
    tag = f"pixel{x}-{y}"
    draw_pixel(canvas, x, y, color, pixel_size, tag)
    print(map)
    
    
def change_to_green(x, y):
    map[y][x] = 2
    color = colors[2]
    tag = f"pixel{x}-{y}"
    draw_pixel(canvas, x, y, color, pixel_size, tag)
    print(map)
    
def place_random_trash():
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    change_to_green(x, y)
    
def place_random_tree():
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    change_to_red(x, y)

canvas_width = 700  # Change this to change the width of the canvas
canvas_height = 700  # Change this to change the height of the canvas
window_size=200
pixel_size = 3*window_size // len(map)
root = tk.Tk()

def reset_grid():
    for y in range(len(map)):
        for x in range(len(map[y])):
            change_to_white(x, y)
x=0
def cycle_trash():
    global x
    if x == 0:
        x = 1
        place_trash_periodically()
    else:
        x = 0
def place_trash_periodically():
    if x == 1:
        place_random_trash()
        root.after(100, place_trash_periodically)  # Schedule next call


   
reset_button = tk.Button(root, text="Réinitialiser", command=reset_grid)
reset_button.pack()
trash_button = tk.Button(root, text="Déchet au hasard", command=place_random_trash)
trash_button.pack()
tree_button = tk.Button(root, text="Tronc au hasard", command=place_random_tree)
tree_button.pack()

trash_cycle_button = tk.Button(root, text="Déchet fréquents", command=cycle_trash)
trash_cycle_button.pack()

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Draw a white background
canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill='white')

draw_map(canvas, window_size)
root.mainloop()