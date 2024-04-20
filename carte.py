import tkinter as tk
from functools import partial
import random
import time
import threading


taille_carte = 30
interval = 1

canvas_width = 700
canvas_height = 700
window_size = 200


class Cellule:
    # ajouter class robot et drone qui ne sont pas reliés à la classe cellule
    def __init__(
        self,
        etage1="void",
        etage2="void",
        date=None,
        robot=False,
        drone=False,
    ):
        self.etage1 = etage1
        self.etage2 = etage2


def init_map(num):
    return [[Cellule() for _ in range(num)] for _ in range(num)]


map = init_map(taille_carte)
pixel_size = 3 * window_size // len(map)

COLORS = {
    "void": "#FFFFFF",
    "obstacle": "#FF0000",
    "trash": "#000000",
    "feuillage": "#CCFFCC",
    "tronc": "#8B4513",
}


###création de blocs / dessin d'une case

clicked_x = clicked_y = None


def draw_pixel(canvas, x, y, color, pixel_size, tag):
    canvas.create_rectangle(
        x * pixel_size,
        y * pixel_size,
        (x + 1) * pixel_size,
        (y + 1) * pixel_size,
        fill=color,
        outline="black",
        tags=tag,
    )


def select_block(start_x, start_y, end_x, end_y, new_color):
    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            map[y][x] = new_color


def draw_map(canvas, window_size):
    pixel_size = 3 * window_size // len(map)
    for y, row in enumerate(map):
        for x, value in enumerate(row):
            color = COLORS[value.etage1]
            tag = f"pixel{x}-{y}"
            draw_pixel(canvas, x, y, color, pixel_size, tag)
            canvas.tag_bind(
                tag, "<Button-1>", lambda event, x=x, y=y: on_click(event, x, y)
            )


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

        menu = tk.Menu(root, tearoff=0)
        menu.add_command(
            label="Changer en obstacle",
            command=lambda: select_and_draw(start_x, start_y, end_x, end_y, 1),
        )
        menu.add_command(
            label="Réinitialiser",
            command=lambda: select_and_draw(start_x, start_y, end_x, end_y, 0),
        )
        menu.post(root.winfo_pointerx(), root.winfo_pointery())
        clicked_x = clicked_y = None


def select_and_draw(start_x, start_y, end_x, end_y, value):
    select_block(start_x, start_y, end_x, end_y, value)
    draw_map(canvas, window_size)


###menu contextuel


def on_click(event, x, y):
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(
        label="Changer en obstacle", command=partial(change_to_obstacle, x, y)
    )
    menu.add_command(label="Changer en arbre", command=partial(change_to_tree, x, y))
    menu.add_command(label="Changer en déchet", command=partial(change_to_trash, x, y))
    menu.add_command(label="Réinitialiser", command=partial(change_to_void, x, y))
    menu.add_command(label="Faire un bloc", command=partial(creation_bloc, x, y))
    menu.post(event.x_root, event.y_root)


###fonctions de changement


# def change_color_1(x, y, element):
#     map[y][x].etage1 = element
#     tag = f"pixel{x}-{y}"
#     draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)


# def change_color_2(x, y, element):
#     map[y][x].etage2 = element
#     tag = f"pixel{x}-{y}"
#     draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)


def change_color(x, y, etage, element):
    if map[y][x].etage2 == "void" and map[y][x].etage1 == "void" and etage == 2:
        print(x, y)
        print(map[y][x].etage1)
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)
        print("1")
    elif map[y][x].etage1 == "void" and map[y][x].etage2 == "void" and etage == 1:
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)
        print("2")

    elif map[y][x].etage2 != "void" and etage == 1:
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, mixed_color(element), pixel_size, tag)
        print("3")

    elif map[y][x].etage1 != "void" and etage == 2:
        print("4")
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"
        print(map[y][x].etage1)
        draw_pixel(canvas, x, y, mixed_color(map[y][x].etage1), pixel_size, tag)
        print("4")


def mixed_color(element):
    COLORS = {
        "trash": "#006400",
        "obstacle": "#8B0000",
        "tronc": "#8B4513",
    }
    return COLORS[element]


def change_to_void(x, y):
    change_color(x, y, 1, "void")
    change_color(x, y, 2, "void")


def change_to_trash(x, y):
    change_color(x, y, 1, "trash")


def change_to_obstacle(x, y):
    change_color(x, y, 1, "obstacle")


def change_to_leaf(x, y):
    change_color(x, y, 2, "feuillage")


def change_to_tree(x, y):
    radius = 2
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:  # Condition pour créer un cercle
                new_x, new_y = x + i, y + j
                if 0 <= new_x < len(map[0]) and 0 <= new_y < len(
                    map
                ):  # Vérifier les limites
                    change_color(new_x, new_y, 2, "feuillage")

    change_color(x, y, 1, "tronc")


###boutons fonctions avancées
def place_random_trash():
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    change_to_trash(x, y)


def place_random_tree():
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    change_to_tree(x, y)


cliqued_switch = 0


def reset_grid():
    for y in range(len(map)):
        for x in range(len(map[y])):
            change_to_void(x, y)


def place_trash_periodically():
    global cliqued_switch
    while cliqued_switch == 1:
        place_random_trash()
        time.sleep(interval)


def cycle_trash():
    global cliqued_switch
    if cliqued_switch == 0:
        cliqued_switch = 1
        threading.Thread(target=place_trash_periodically).start()
    else:
        cliqued_switch = 0


root = tk.Tk()
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

canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="white")

draw_map(canvas, window_size)
root.mainloop()
