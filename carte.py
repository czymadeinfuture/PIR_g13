import random
import threading
import time
import tkinter as tk
from functools import partial

from colorama import Fore, Style

###################################################
# initialisation
###################################################

taille_carte = 5
interval_spawn_dechet = 1

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


def init_map(num):  # initialisation de la carte
    return [[Cellule() for _ in range(num)] for _ in range(num)]


map = init_map(taille_carte)
pixel_size = 3 * window_size // len(map)

COLORS = {  # couleurs des cases
    "void": "#FFFFFF",
    "obstacle": "#FF0000",
    "trash": "#000000",
    "feuillage": "#00B500",
    "tronc": "#8B4513",
}


###################################################
###création de blocs / dessin d'une case
###################################################

clicked_x = clicked_y = None


def draw_pixel(canvas, x, y, color, pixel_size, tag):  # dessin d'un pixel normal
    canvas.create_rectangle(
        x * pixel_size,
        y * pixel_size,
        (x + 1) * pixel_size,
        (y + 1) * pixel_size,
        fill=color,
        outline="black",
        tags=tag,
    )


def draw_pixel_feuillage(
    canvas, x, y, color, pixel_size, tag
):  # dessin d'un pixel sous un feuillage
    draw_pixel(canvas, x, y, "#00B500", pixel_size, tag)
    margin = pixel_size // 6
    canvas.create_rectangle(
        x * pixel_size + margin,
        y * pixel_size + margin,
        (x + 1) * pixel_size - margin,
        (y + 1) * pixel_size - margin,
        fill=color,
        tags=tag,
    )


def draw_map(canvas, window_size):  # dessin de la carte
    pixel_size = 3 * window_size // len(map)
    for y, row in enumerate(map):
        for x, value in enumerate(row):
            color = COLORS[value.etage1]
            tag = f"pixel{x}-{y}"
            change_color(x, y, 1, value.etage1)
            canvas.tag_bind(
                tag, "<Button-1>", lambda event, x=x, y=y: on_click(event, x, y)
            )


def select_block(
    start_x, start_y, end_x, end_y, new_etage1
):  # changement d'un block dans la matrice
    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            change_color(x, y, 1, new_etage1)


def select_and_draw(
    start_x, start_y, end_x, end_y, etage1_value
):  # dessin de la carte après un bloc
    select_block(start_x, start_y, end_x, end_y, etage1_value)
    # draw_map(canvas, window_size)


def creation_bloc(x, y):  # création d'un bloc
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
            command=lambda: select_and_draw(start_x, start_y, end_x, end_y, "obstacle"),
        )
        menu.add_command(
            label="Réinitialiser",
            command=lambda: select_and_draw(start_x, start_y, end_x, end_y, "void"),
        )
        menu.post(root.winfo_pointerx(), root.winfo_pointery())
        clicked_x = clicked_y = None


###################################################
###fonctions des pixels (principal) (comportement des pixels)
###################################################


def on_click(event, x, y):  # menu contextuel au clic sur un pixel
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(
        label="Changer en obstacle", command=partial(change_to_obstacle, x, y)
    )
    menu.add_command(label="Changer en arbre", command=partial(change_to_tree, x, y))
    menu.add_command(label="Changer en déchet", command=partial(change_to_trash, x, y))
    menu.add_command(label="Réinitialiser", command=partial(change_to_void, x, y))
    menu.add_command(label="Faire un bloc", command=partial(creation_bloc, x, y))
    menu.post(event.x_root, event.y_root)


def change_color(
    x, y, etage, element
):  # changement de couleur d'un pixel, en fonction des elements des étages
    if map[y][x].etage1 == "tronc" and etage == 1 and element != "tronc":
        tag = f"pixel{x}-{y}"
        map[y][x].etage1 = element

        radius = 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i**2 + j**2 <= radius**2:
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < len(map[0]) and 0 <= new_y < len(map):
                        tag_new = f"pixel{new_x}-{new_y}"
                        if (
                            map[new_y][new_x].etage1 != "void"
                            and map[new_y][new_x].etage1 != "tronc"
                        ):
                            map[new_y][new_x].etage2 = "void"
                            draw_pixel(
                                canvas,
                                new_x,
                                new_y,
                                COLORS[map[new_y][new_x].etage1],
                                pixel_size,
                                tag_new,
                            )
                        elif (
                            map[new_y][new_x].etage1 != "void"
                            and map[new_y][new_x].etage1 == "tronc"
                        ):
                            map[new_y][new_x].etage2 = "feuillage"
                            draw_pixel_feuillage(
                                canvas,
                                new_x,
                                new_y,
                                COLORS[map[new_y][new_x].etage1],
                                pixel_size,
                                tag_new,
                            )
                        else:
                            map[new_y][new_x].etage2 = "void"
                            draw_pixel(
                                canvas,
                                new_x,
                                new_y,
                                COLORS["void"],
                                pixel_size,
                                tag_new,
                            )

        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)

    elif map[y][x].etage2 == "void" and map[y][x].etage1 == "void" and etage == 2:
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)

    elif map[y][x].etage1 == "void" and map[y][x].etage2 == "void" and etage == 1:
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)

    elif map[y][x].etage2 != "void" and etage == 1 and element != "void":
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel_feuillage(canvas, x, y, COLORS[element], pixel_size, tag)

    elif map[y][x].etage1 != "void" and etage == 2:
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"
        draw_pixel_feuillage(canvas, x, y, COLORS[map[y][x].etage1], pixel_size, tag)

    elif map[y][x].etage2 != "void" and etage == 1 and element == "void":
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS["feuillage"], pixel_size, tag)

    elif map[y][x].etage1 != "void" and element == "void":
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS["void"], pixel_size, tag)


def change_to_void(x, y):  # changement d'un pixel en void
    change_color(x, y, 1, "void")


def change_to_trash(x, y):  # changement d'un pixel en déchet
    change_color(x, y, 1, "trash")


def change_to_obstacle(x, y):  # changement d'un pixel en obstacle
    change_color(x, y, 1, "obstacle")


def change_to_leaf(x, y):  # changement d'un pixel en feuillage
    change_color(x, y, 2, "feuillage")


def change_to_tree(x, y):  # changement d'un pixel en arbre (tronc + feuillage)
    radius = 2
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                new_x, new_y = x + i, y + j
                if 0 <= new_x < len(map[0]) and 0 <= new_y < len(map):
                    change_color(new_x, new_y, 2, "feuillage")

    change_color(x, y, 1, "tronc")


###################################################
###boutons fonctions avancées
###################################################


def print_map(map):  # affichage de la matrice dans la console
    color_mapping = {
        "feuillage": Fore.GREEN,
        "tronc": Fore.YELLOW,
        "obstacle": Fore.RED,
        "dechet": Fore.BLUE,
        "void": Fore.WHITE,
    }
    print("Matrice :")
    print("   ")
    for row in map:
        for cell in row:
            etage1_color = color_mapping.get(cell.etage1, Fore.RESET)
            etage2_color = color_mapping.get(cell.etage2, Fore.RESET)
            print(
                f"{etage1_color}({cell.etage1}, {etage2_color}{cell.etage2}){Style.RESET_ALL}",
                end=" ",
            )

        print()
    print("   ")
    print("   ")


def place_random_trash():  # fonction placement d'un déchet aléatoire
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    if map[y][x].etage1 == "void":
        change_to_trash(x, y)


def place_random_tree():  # fonction placement d'un arbre aléatoire
    x = random.randint(0, len(map) - 1)
    y = random.randint(0, len(map[0]) - 1)
    change_to_tree(x, y)


cliqued_switch = 0


def reset_grid():  # réinitialisation de la grille
    for y in range(len(map)):
        for x in range(len(map[y])):
            change_to_void(x, y)


def place_trash_periodically():  # fonction placement de déchets périodique
    global cliqued_switch
    while cliqued_switch == 1:
        place_random_trash()
        time.sleep(interval_spawn_dechet)


def cycle_trash():  # fonction pour activer/désactiver le placement de déchets périodique
    global cliqued_switch
    if cliqued_switch == 0:
        cliqued_switch = 1
        threading.Thread(target=place_trash_periodically).start()
    else:
        cliqued_switch = 0


###################################################
###interface graphique
###################################################


def bouton(
    nom_bouton, commande_bouton, couleur="green"
):  # création d'un bouton, couleur en paramètre
    return tk.Button(
        root,
        text=nom_bouton,
        command=commande_bouton,
        fg="white",
        bg=couleur,
        font=("Helvetica", 10),
    )


root = tk.Tk()

trash_button = bouton("Déchet au hasard", place_random_trash, "black")
trash_button.grid(row=0, column=0, sticky="w")

tree_button = bouton("Tronc au hasard", place_random_tree, "green")
tree_button.grid(row=1, column=0, sticky="w")

trash_cycle_button = bouton("Déchet fréquents", cycle_trash, "black")
trash_cycle_button.grid(row=2, column=0, sticky="w")

tree_button = bouton("Afficher matrice dans console", lambda: print_map(map), "blue")
tree_button.grid(row=0, column=1, sticky="e")

reset_button = bouton("Réinitialiser", reset_grid, "red")
reset_button.grid(row=1, column=1, sticky="e")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.grid(row=3, column=0, columnspan=2)  # Use grid here

canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="white")

draw_map(canvas, window_size)
root.mainloop()
