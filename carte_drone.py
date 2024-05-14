import random
import subprocess
import threading
import time
import tkinter as tk
from functools import partial
import tkinter.simpledialog as sd
import tkinter as tk
from tkinter.simpledialog import Dialog
import math
import pulp
import numpy as np
from geopy import distance as dis
from colorama import Fore, Style
import copy
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.exact import solve_tsp_dynamic_programming


###################################################
# initialisation
###################################################

taille_carte = 18
interval_spawn_dechet = 1

canvas_width = 900
canvas_height = 900
window_size = 270

# element pour des drones
dt = 1
taille_section = taille_carte // 2
taille_bloc = 3
nb = taille_section//taille_bloc

#
lines = []

class Cellule:
    def __init__(
        self,
        etage1="void",
        etage2="void",
        date=None,  # Permettra plus tard de definir la derniere fois que la case a ete visitee
        robot=False,
        drone=False,
    ):
        self.etage1 = etage1
        self.etage2 = etage2
        self.robot = Robot() if robot else None
        self.drone = Drone() if drone else None


class Robot:
    def __init__(self, x=0, y=0, quantite=0, capacite=25):
        self.x = (
            x  # Permettra plus tard les positions relatives dans la cellule sur Gazebo
        )
        self.y = y
        self.quantite = quantite  # Permettra plus tard de définir le nombre de déchet que le robot a en stock
        self.capacite = capacite  # Permettra plus tard de définir la capacité max de déchet que le robot peut avoir


class Drone:
    def __init__(self, x=0, y=0, z=0):
        self.x = (
            x  # Permettra plus tard les positions relatives dans la cellule sur Gazebo
        )
        self.y = y
        self.z = z

class Drone_courant:
    def __init__(self,start_x, end_x, start_y, end_y, id, canvas):

        self.id = id

        # positions possibles pour chaque drone dans san zone 
        self.list = self.init_list(start_x,start_y)

        # noter les points qui sont visites par chaque drone
        self.waypoints= []

        # zone de detection
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y

        self.canvas = canvas

        # parametre pour le scan
        self.scan_radius = int(pixel_size * 1.2)

        # taille de drone
        self.drone_size =  pixel_size * 0.2 

        # position initiale
        self.x, self.y = self.init_position()

        self.target_x, self.target_y = self.x, self.y
        self.moving = False

        self.path = []

        # une matrice pour stocker le nb de visite
        self.visited = [[0]*nb for _ in range(nb)]

        # une liste pour stocker les positions de dechets
        self.trash_found = []

        # registrer seulement les positions des arebres et des obstacles dans la matrice de map
        self.map_copy= [[0] * taille_carte for _ in range(taille_carte)]  
        self.init_map_copy()

        # une matrice pour stocker l'heure de la derniere visite
        self.visited_time = [[0]*taille_carte for _ in range(taille_carte)]

        # la direction du mouvement de la drone
        self.last_direction = None 

        self.min_visit = 0

        # parametre pour controler le statue du mouvement
        self.paused = False

        # parametre pour controler la ferme de le processus
        self.should_run = True

        self.lock = threading.Lock() 


    def initial_scan(self, canvas):
        self.scan_for_trash(canvas)

    def init_map_copy(self):
        for y in range(self.start_y, self.end_y + 1):
            for x in range(self.start_x, self.end_x + 1):
                if map_data[y][x] in (1, 2):  
                    self.map_copy[y][x] = map_data[y][x]

    def all_covered(self):
        for y in range(nb):
            for x in range(nb):
                if self.visited[y][x] == self.min_visit:
                    return False
        return True

    def init_list(self,start_x,start_y):
        list = []
        for i in range(nb):
            for j in range(nb):
                list.append((start_x+i*taille_bloc+1,start_y+j*taille_bloc+1))
        return list

    def init_position(self):
        x,y = random.choice(self.list)
        self.waypoints.append((x,y))
        return x,y

    
    def draw_drone(self, canvas):
        tag = f"drone{self.start_x}-{self.start_y}"
        center_x = (self.x + 0.5) * pixel_size
        center_y = (self.y + 0.5) * pixel_size
        self.path.append((center_x, center_y))
        canvas.delete(tag)

        # dessine de drones
        margin = pixel_size // 2.5
        radius = pixel_size // 3.5
        for dx in [0.15, 0.85]:
            for dy in [0.15, 0.85]:
                canvas.create_oval(
                    (self.x + dx) * pixel_size - radius + margin,
                    (self.y + dy) * pixel_size - radius + margin,
                    (self.x+ dx) * pixel_size + radius - margin,
                    (self.y + dy) * pixel_size + radius - margin,
                    fill="#FFB200",
                    tags=tag,
                )

        # dessine de zone scanne
        canvas.create_oval(
            center_x - self.scan_radius, center_y - self.scan_radius,
            center_x + self.scan_radius, center_y + self.scan_radius,
            outline="#FFD700", width=2, tags=tag
        )


    def scan_for_trash(self, canvas):
        radius_pixels = self.scan_radius 
        center_x = (self.x + 0.5) * pixel_size
        center_y = (self.y + 0.5) * pixel_size 

        for dx in range(-radius_pixels, radius_pixels + 1):
            for dy in range(-radius_pixels, radius_pixels + 1):
                if dx**2 + dy**2 < radius_pixels**2:  
                    pixel_x = center_x + dx
                    pixel_y = center_y + dy
                    grid_x = int((pixel_x // pixel_size))
                    grid_y = int((pixel_y // pixel_size))
                    if 0 <= grid_x < taille_carte and 0 <= grid_y < taille_carte:
                        item_id = canvas.find_closest(pixel_x, pixel_y)
                        item_color = canvas.itemcget(item_id, "fill")
                        if self.map_copy[grid_y][grid_x] == 0 and item_color == COLORS["trash"]:
                            if (grid_x, grid_y) not in self.trash_found:  
                                self.trash_found.append((grid_x, grid_y))
                                print(f"DRONE {self.id} finds trash at position: ({grid_x+1}, {grid_y+1})")

    # utiliser algo nc-drone-ts pour décider la position suivante
    def start_move(self):
        if not self.moving:
            min_visits = float('inf')
            best_directions = []
            directions = {
                'up': (0, -3),
                'down': (0, 3),
                'left': (-3, 0),
                'right': (3, 0)
            }

            for direction, (dx, dy) in directions.items():
                nx, ny = self.x + dx, self.y + dy
                if self.start_x <= nx <= self.end_x and self.start_y <= ny <= self.end_y:  
                    visits = self.visited[(ny-self.start_y)//taille_bloc][(nx-self.start_x)//taille_bloc]
                    last_time = self.visited_time[ny][nx]
                    nb_obstacles = block_counts[ny // taille_bloc][nx // taille_bloc]

                    if (visits < min_visits or 
                        (visits == min_visits and nb_obstacles < min_obstacles) or
                        (visits == min_visits and nb_obstacles == min_obstacles and last_time < oldest_time)):
                        min_visits = visits
                        min_obstacles = nb_obstacles
                        oldest_time = last_time
                        best_directions = [direction]
                    elif visits == min_visits and nb_obstacles == min_obstacles and last_time == oldest_time:
                        best_directions.append(direction)

            if best_directions:
                if self.last_direction in best_directions:
                    best_direction = self.last_direction
                else:
                    best_direction = random.choice(best_directions)  

                self.last_direction = best_direction

            if best_direction == "up" and self.y > 0:
                self.target_y -= taille_bloc
            elif best_direction == "down" and self.y < taille_carte - 1:
                self.target_y += taille_bloc
            elif best_direction == "left" and self.x > 0:
                self.target_x -= taille_bloc
            elif best_direction == "right" and self.x < taille_carte - 1:
                self.target_x += taille_bloc
            else:
                return  
            self.vx = 0.20 if self.x < self.target_x else -0.20 if self.x > self.target_x else 0
            self.vy = 0.20 if self.y < self.target_y else -0.20 if self.y > self.target_y else 0
            self.moving = True
            self.visited[(self.y-self.start_y)// taille_bloc][(self.x-self.start_x) // taille_bloc] += 1
            self.visited_time[self.y][self.x]= time.time()
            self.waypoints.append((self.target_x,self.target_y))

    # faire le mouvement
    def move_to_target(self, canvas): 
        if self.moving:
            if (round(self.x, 1) != self.target_x) or (round(self.y, 1) != self.target_y):
                self.x += self.vx * dt
                self.y += self.vy * dt
            else:
                self.x = self.target_x
                self.y = self.target_y
                self.moving = False                
                self.scan_for_trash(canvas) 

            self.draw_drone(canvas)
            root.after(10, lambda: self.move_to_target(canvas))
            self.update_gui()
    
    def pause(self):
        with self.lock:
            self.paused = True

    def restart(self):
        with self.lock:
            if self.paused:
                self.paused = False
                if not self.moving:
                    threading.Thread(target=self.run).start()

    def run(self):
        self.initial_scan(self.canvas)
        while self.should_run and not self.paused:
            if not self.moving:
                self.start_move()
                self.move_to_target(self.canvas)
                if self.all_covered():
                    self.min_visit += 1
                    time.sleep(2)  
                else:
                    time.sleep(0.2) 
            else:
                time.sleep(0.2) 
            if not self.should_run: 
                break

    #arreter le processus
    def stop(self):
        self.should_run = False

    def update_gui(self):
        self.canvas.after(0, self.draw_drone, self.canvas)


def init_map(num):  # initialisation de la carte
    return [[Cellule() for _ in range(num)] for _ in range(num)]

# ----------------- todo ----------
# Il faut faire la connaissance de la carte (positions des obstacle et des arbres)
def init_map_drone(num):
    map_data = [[0]*num for _ in range(num)]

    #ajouter des positions des obstacle et des arbres
    
    block_counts = [[0] * (taille_carte//taille_bloc) for _ in range(taille_carte//taille_bloc)]
    for y in range(taille_carte//taille_bloc):
        for x in range(taille_carte//taille_bloc):
            count = sum(1 for dy in range(taille_bloc)
                        for dx in range(taille_bloc)
                        if map_data[y * taille_bloc + dy][x * taille_bloc + dx] in {1, 2})
            block_counts[y][x] = count

    return map_data,block_counts


##################################################

map = init_map(taille_carte)
map_data,block_counts = init_map_drone(taille_carte)
pixel_size = 3 * window_size // len(map)

COLORS = {  # couleurs des cases
    "void": "#FFFFFF",
    "obstacle": "#FF0000",
    "trash": "#000000",
    "feuillage": "#00B500",
    "tronc": "#8B4513",
    "robot": "#0000FF",
    "drone": "#FFB200",
    "blue": "#0000FF",
}


###################################################
### pour les drones
###################################################

def draw_drones(canvas):
    for drone in drones:
        drone.draw_drone(canvas)


def pause_drones():
    for drone in drones:
        drone.pause()

def start_drones():
    for drone in drones:
        if not drone.moving and not drone.paused:
            drone_thread = threading.Thread(target=drone.run)
            drone_thread.start()
        elif drone.paused:
            drone.restart()

def stop_all_drones_and_exit():
    for drone in drones:
        drone.stop() 
    root.destroy() 



###################################################
### mTSP
###################################################


# 2 fonctions pr le tsp
def eucl_distance(a, b):
    distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return round(distance)


def tsp_calculation(nodes_matrix):
    distance_matrix = []
    for i in range(len(nodes_matrix)):
        distance_matrix.append([])
        for e in range(len(nodes_matrix)):
            distance_2_points = eucl_distance(nodes_matrix[i], nodes_matrix[e])
            distance_matrix[i].append(distance_2_points)

    distance_matrix_np = np.array(distance_matrix)

    # print(distance_matrix_np)

    # pr changer de méthode d'approche pr le tsp, changer sole_tsp_dynamic_programming par autre chose
    # par exemplesole_tsp_local_search (voir le github pr ttes les méthodes)
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix_np)

    # print(permutation, "\n", distance)

    return permutation, distance


def bresenham_line(x0, y0, x1, y1):
    """
    Implémente l'algorithme de Bresenham pour tracer une ligne entre deux points.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        points.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    points.append((x1, y1))
    return points


def has_obstacle(matrix, x0, y0, x1, y1):
    """
    Vérifie la présence d'obstacles entre deux points dans une matrice 2D.
    """
    line = bresenham_line(x0, y0, x1, y1)
    for x, y in line:
        if matrix[x][y].etage1 == "obstacle":  # Obstacle dans la matrice
            return True, (x, y)  # Retourne True et les coordonnées de l'obstacle
    return (
        False,
        None,
    )  # Si aucun obstacle n'est rencontré, retourne False et None pour les coordonnées de l'obstacle


def mTSP():
    indices_dechets = []
    indices_robots = []
    trajet = []
    couleurs_robots = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "black",
        "yellow",
    ]

    for i in range(taille_carte):
        for j in range(taille_carte):
            if map[i][j].robot is not None:
                indices_robots.append((i, j))
                trajet.append((i, j))
    for i in range(taille_carte):
        for j in range(taille_carte):
            if map[i][j].etage1 == "trash":  # Si la cellule contient des déchets
                indices_dechets.append((i, j))  # Enregistrer l'indice de la cellule
                trajet.append((i, j))
            if map[i][j].etage1 == "tronc":
                if j != 0:
                    indices_dechets.append((i, j - 1))
                    trajet.append((i, j - 1))
                else:
                    indices_dechets.append((i, j + 1))
                    trajet.append((i, j + 1))
    if len(indices_dechets) >= 2 and len(indices_robots) >= 1:
        ################################################
        # Building distance matrix
        ################################################
        n = len(trajet)
        C = np.zeros((n, n))
        print(len(trajet))
        for i in range(0, n):
            for j in range(0, len(trajet)):
                C[i, j] = dis.distance(trajet[i], trajet[j]).m

        distance_matrix = np.array(C)
        num_robots = len(indices_robots)
        num_debris = len(indices_dechets)
        print(num_robots)
        print(num_debris)
        # Création du problème
        prob = pulp.LpProblem("mTSP", pulp.LpMinimize)

        # Déclaration des variables
        x = pulp.LpVariable.dicts(
            "x",
            [(i, j) for i in range(num_robots) for j in range(num_robots, len(trajet))],
            cat="Binary",
        )

        # Fonction objectif
        prob += pulp.lpSum(
            distance_matrix[i, j] * x[(i, j)]
            for i in range(num_robots)
            for j in range(num_robots, len(trajet))
        )

        # Contraintes
        for j in range(num_robots, len(trajet)):
            prob += (
                pulp.lpSum(x[(i, j)] for i in range(num_robots)) == 1
            )  # Chaque déchet est attribué à exactement un robot

        # Résolution
        prob.solve()

        # Affichage de la solution
        trajet_par_robot = {}
        print(f"Trajet : {trajet}")
        for i in range(num_robots):
            print(f"Chemin pour le robot {i+1}:")
            trajet_par_robot[i + 1] = []
            for j in range(num_robots, len(trajet)):
                if pulp.value(x[(i, j)]) == 1:
                    trajet_par_robot[i + 1].append(trajet[j])
                    print(f"Déchet {j+1-num_robots}")
        print(trajet_par_robot)

        print(
            "Distance totale minimale parcourue par tous les robots :",
            pulp.value(prob.objective),
        )

        trajet_par_robot_tsp = copy.deepcopy(trajet_par_robot)
        for robot in trajet_par_robot:
            nodes_matrix = trajet_par_robot[robot]
            if len(nodes_matrix) != 0:
                permutation, distance = tsp_calculation(nodes_matrix)
                print(
                    f"\nRobot {robot} doit parcourir les points {permutation} pour une distance de {distance}"
                )
                for i in range(len(permutation)):
                    print(permutation[i])
                    # print(f"trajet_par_robot = {trajet_par_robot[robot]}")
                    point_equivalent = trajet_par_robot[robot][permutation[i]]
                    # print(point_equivalent)
                    trajet_par_robot_tsp[robot][i] = trajet_par_robot[robot][
                        permutation[i]
                    ]
        print(f"\nTrajet final par robot : {trajet_par_robot_tsp}")

        # afficher les trajets
        print(f"lines : {lines}")
        for line in lines:
            canvas.delete(line)
        for robot in trajet_par_robot_tsp:
            ancien_point = trajet[robot - 1]
            for point in trajet_par_robot_tsp[robot]:
                print(ancien_point, point)
                line = canvas.create_line(
                    (1 / 2 + float(ancien_point[1])) * pixel_size,
                    (1 / 2 + float(ancien_point[0])) * pixel_size,
                    (1 / 2 + float(point[1])) * pixel_size,
                    (1 / 2 + float(point[0])) * pixel_size,
                    fill=couleurs_robots[robot - 1],
                    width=3,
                )
                lines.append(line)
                ancien_point = point
        # resolution du TSP par robot


###################################################
###Gazebo
###################################################


def gazebo(type_obj, y, x):
    if type_obj == "intelaero":
        command = f"cd ~/catkin_ws && source devel/setup.bash && roslaunch gazebo_project intelaero.launch drone_name:={type_obj}_{x}_{y} x:={-(x-taille_carte//2)} y:={-(y-taille_carte//2)} z:=10 roll:=0.0 pitch:=0.0 yaw:=0.0 detector_range:=10"  # noqa: E501
    else:
        command = f"cd ~/catkin_ws && source devel/setup.bash && roslaunch gazebo_project {type_obj}.launch drone_name:={type_obj}_{x}_{y} x:={-(x-taille_carte//2)} y:={-(y-taille_carte//2)}"  # noqa: E501
    try:
        subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to launch Gazebo: {e}")


def gazebo_delete(type_obj, y=None, x=None):
    model_name = f"{type_obj}" if x is None and y is None else f"{type_obj}_{x}_{y}"
    command = f"cd ~/catkin_ws && source devel/setup.bash && rosservice call gazebo/delete_model '{{model_name: {model_name}}}'"  # noqa: E501
    print(command)
    try:
        subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to delete Gazebo model: {e}")


def gazebo_deplacer(type_obj, y, x):
    command = f"cd ~/catkin_ws && source devel/setup.bash && rosservice call gazebo/delete_model '{{model_name: {type_obj}_{x}_{y}}}'"  # noqa: E501
    try:
        subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to delete Gazebo model: {e}")


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


def draw_pixel_drone(
    canvas, x, y, color, pixel_size, tag
):  # dessin d'un pixel sous un drone
    margin = pixel_size // 2.5
    radius = pixel_size // 3.5
    for dx in [0.15, 0.85]:
        for dy in [0.15, 0.85]:
            canvas.create_oval(
                (x + dx) * pixel_size - radius + margin,
                (y + dy) * pixel_size - radius + margin,
                (x + dx) * pixel_size + radius - margin,
                (y + dy) * pixel_size + radius - margin,
                fill=color,
                tags=tag,
            )


def draw_pixel_robot(
    canvas, x, y, color, pixel_size, tag
):  # dessin d'un pixel sous un robot
    margin = pixel_size // 4
    width = pixel_size // 10
    # Draw black outline
    canvas.create_line(
        x * pixel_size + margin,
        y * pixel_size + margin,
        (x + 1) * pixel_size - margin,
        (y + 1) * pixel_size - margin,
        fill="black",
        width=width * 2,
        tags=tag,
    )
    canvas.create_line(
        (x + 1) * pixel_size - margin,
        y * pixel_size + margin,
        x * pixel_size + margin,
        (y + 1) * pixel_size - margin,
        fill="black",
        width=width * 2,
        tags=tag,
    )
    # Draw colored line
    canvas.create_line(
        x * pixel_size + margin,
        y * pixel_size + margin,
        (x + 1) * pixel_size - margin,
        (y + 1) * pixel_size - margin,
        fill=color,
        width=width,
        tags=tag,
    )
    canvas.create_line(
        (x + 1) * pixel_size - margin,
        y * pixel_size + margin,
        x * pixel_size + margin,
        (y + 1) * pixel_size - margin,
        fill=color,
        width=width,
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
            if new_etage1 == "obstacle":
                gazebo("obstacle", x, y)
            elif new_etage1 == "void":
                print(f"Deleting {map[y][x].etage1} at ({x}, {y})")
                if map[y][x].etage1 == "trash":
                    gazebo_delete("trash", x, y)
                elif map[y][x].etage1 == "tree":
                    gazebo_delete("tree", x, y)
                elif map[y][x].etage1 == "obstacle":
                    gazebo_delete("obstacle", x, y)


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

    change_menu = tk.Menu(menu, tearoff=0)
    change_menu.add_command(
        label="Changer en obstacle", command=partial(change_to_obstacle, x, y)
    )
    change_menu.add_command(
        label="Changer en arbre", command=partial(change_to_tree, x, y)
    )
    change_menu.add_command(
        label="Changer en déchet", command=partial(change_to_trash, x, y)
    )

    robot_menu = tk.Menu(menu, tearoff=0)
    robot_menu.add_command(
        label="Placer un robot", command=partial(change_to_robot, x, y)
    )
    robot_menu.add_command(
        label="Placer un drone", command=partial(change_to_drone, x, y)
    )
    if map[y][x].drone is not None:
        robot_menu.add_command(
            label="Déplacer robot", command=partial(animate_drone_move, x, y)
        )
        robot_menu.add_command(
            label="Retirer drone", command=partial(retirer_drone, x, y)
        )
    if map[y][x].robot is not None:
        robot_menu.add_command(
            label="Retirer robot", command=partial(retirer_robot, x, y)
        )

    menu.add_cascade(label="Changer en", menu=change_menu)
    menu.add_cascade(label="Robots", menu=robot_menu)
    if map[y][x].etage1 != "void":
        menu.add_command(label="Supprimer", command=partial(change_to_void, x, y))

    menu.add_command(label="Faire un bloc", command=partial(creation_bloc, x, y))

    menu.tk_popup(event.x_root, event.y_root)


class CoordinateDialog(Dialog):
    def body(self, master):
        tk.Label(master, text="x:").grid(row=0)
        tk.Label(master, text="y:").grid(row=1)

        self.x_entry = tk.Entry(master)
        self.y_entry = tk.Entry(master)

        self.x_entry.grid(row=0, column=1)
        self.y_entry.grid(row=1, column=1)

        return self.x_entry  # initial focus

    def apply(self):
        self.result = (int(self.x_entry.get()), int(self.y_entry.get()))


def ask_for_coordinates():
    dialog = CoordinateDialog(root)
    return dialog.result


def retirer_drone(x, y):
    if map[y][x].drone is not None:
        map[y][x].drone = None
        try:
            gazebo_delete("intelaero", x, y)
        except Exception as e:
            print(f"Failed to delete drone: {e}")
        change_color(x, y, 1, map[y][x].etage1)
        if map[y][x].etage2 != "void" and map[y][x].etage1 != "void":
            draw_pixel_feuillage(
                canvas, x, y, COLORS[map[y][x].etage1], pixel_size, f"pixel{x}-{y}"
            )
        elif map[y][x].etage1 != "void":
            draw_pixel(canvas, x, y, COLORS["feuillage"], pixel_size, f"pixel{x}-{y}")


def retirer_robot(x, y):
    if map[y][x].robot is not None:
        map[y][x].robot = None
        try:
            gazebo_delete("warthog", x, y)
        except Exception as e:
            print(f"Failed to delete robot: {e}")
        change_color(x, y, 1, map[y][x].etage1)
        if map[y][x].etage2 != "void" and map[y][x].etage1 != "void":
            draw_pixel_feuillage(
                canvas, x, y, COLORS[map[y][x].etage1], pixel_size, f"pixel{x}-{y}"
            )
        elif map[y][x].etage1 != "void":
            draw_pixel(canvas, x, y, COLORS["feuillage"], pixel_size, f"pixel{x}-{y}")

        gazebo_delete("warthog", x, y)
        change_color(x, y, 1, map[y][x].etage1)
        if map[y][x].etage2 != "void":
            draw_pixel_feuillage(
                canvas, x, y, COLORS[map[y][x].etage1], pixel_size, f"pixel{x}-{y}"
            )


def animate_drone_move_step(i, old_x, old_y, new_x, new_y, line=None):
    distance = math.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
    steps = int(distance) * 10

    dx = (new_x - old_x) / steps
    dy = (new_y - old_y) / steps

    x = old_x + dx * i
    y = old_y + dy * i

    previous_x = old_x + dx * (i - 1)
    previous_y = old_y + dy * (i - 1)
    retirer_drone(round(previous_x), round(previous_y))

    draw_pixel_drone(
        canvas,
        round(x),
        round(y),
        COLORS["drone"],
        pixel_size,
        f"pixel{round(x)}-{round(y)}",
    )

    map[round(y)][round(x)].drone = Drone(round(x), round(y))

    if i < steps:
        canvas.after(
            20,
            animate_drone_move_step,
            i + 1,
            old_x,
            old_y,
            new_x,
            new_y,
            line,
        )
    else:
        if line is None:
            line = canvas.create_line(
                (1 / 2 + old_x) * pixel_size,
                (1 / 2 + old_y) * pixel_size,
                (1 / 2 + new_x) * pixel_size,
                (1 / 2 + new_y) * pixel_size,
                fill=COLORS["drone"],
            )


def animate_drone_move(x, y):
    coordinates = ask_for_coordinates()
    if coordinates is not None:
        new_x, new_y = coordinates
        animate_drone_move_step(0, x, y, new_x, new_y)


def change_color(x, y, etage, element):
    if element == "drone":
        draw_pixel_drone(canvas, x, y, COLORS["drone"], pixel_size, f"pixel{x}-{y}")
        map[y][x].drone = Drone(x, y)
    elif element == "robot":
        draw_pixel_robot(canvas, x, y, COLORS["robot"], pixel_size, f"pixel{x}-{y}")
        map[y][x].robot = Robot(x, y)

    elif map[y][x].etage1 == "tronc" and etage == 1 and element != "tronc":
        gazebo_delete("tree", x, y)
        tag = f"pixel{x}-{y}"
        map[y][x].etage1 = element

        radius = 5
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
                            if map[new_y][new_x].drone is not None:
                                draw_pixel_drone(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["drone"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
                                )
                            elif map[new_y][new_x].robot is not None:
                                draw_pixel_robot(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["robot"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
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
                            if map[new_y][new_x].drone is not None:
                                draw_pixel_drone(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["drone"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
                                )
                            elif map[new_y][new_x].robot is not None:
                                draw_pixel_robot(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["robot"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
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
                            if map[new_y][new_x].drone is not None:
                                draw_pixel_drone(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["drone"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
                                )
                            elif map[new_y][new_x].robot is not None:
                                draw_pixel_robot(
                                    canvas,
                                    new_x,
                                    new_y,
                                    COLORS["robot"],
                                    pixel_size,
                                    f"pixel{new_x}-{new_y}",
                                )
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)

    elif (
        map[y][x].etage2 == "void"
        and map[y][x].etage1 == "void"
        and etage == 2
        and element != "drone"
    ):
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)
        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        elif map[y][x].robot is not None:
            tag = f"pixel{x}-{y}"
            draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)
            if map[y][x].robot is not None:
                draw_pixel_robot(
                    canvas,
                    x,
                    y,
                    COLORS["robot"],
                    pixel_size,
                    f"pixel{x}-{y}",
                )

    elif map[y][x].etage1 == "void" and map[y][x].etage2 == "void" and etage == 1:
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS[element], pixel_size, tag)
        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        if map[y][x].robot is not None:
            draw_pixel_robot(
                canvas,
                x,
                y,
                COLORS["robot"],
                pixel_size,
                f"pixel{x}-{y}",
            )

    elif map[y][x].etage2 != "void" and etage == 1 and element != "void":
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel_feuillage(canvas, x, y, COLORS[element], pixel_size, tag)
        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        if map[y][x].robot is not None:
            draw_pixel_robot(
                canvas,
                x,
                y,
                COLORS["robot"],
                pixel_size,
                f"pixel{x}-{y}",
            )

    elif map[y][x].etage1 != "void" and etage == 2:
        map[y][x].etage2 = element
        tag = f"pixel{x}-{y}"

        draw_pixel_feuillage(canvas, x, y, COLORS[map[y][x].etage1], pixel_size, tag)
        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        if map[y][x].robot is not None:
            draw_pixel_robot(
                canvas,
                x,
                y,
                COLORS["robot"],
                pixel_size,
                f"pixel{x}-{y}",
            )

    elif map[y][x].etage2 == "feuillage" and etage == 1 and element == "void":
        if map[y][x].etage1 == "trash":
            gazebo_delete("trash", x, y)
        if map[y][x].etage1 == "obstacle":
            gazebo_delete("obstacle", x, y)
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS["feuillage"], pixel_size, tag)

        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        if map[y][x].robot is not None:
            draw_pixel_robot(
                canvas,
                x,
                y,
                COLORS["robot"],
                pixel_size,
                f"pixel{x}-{y}",
            )

    elif map[y][x].etage1 != "void" and element == "void":
        if map[y][x].etage1 == "trash":
            gazebo_delete("trash", x, y)
        if map[y][x].etage1 == "obstacle":
            gazebo_delete("obstacle", x, y)
        map[y][x].etage1 = element
        tag = f"pixel{x}-{y}"
        draw_pixel(canvas, x, y, COLORS["void"], pixel_size, tag)
        if map[y][x].drone is not None:
            draw_pixel_drone(
                canvas,
                x,
                y,
                COLORS["drone"],
                pixel_size,
                f"pixel{x}-{y}",
            )
        if map[y][x].robot is not None:
            draw_pixel_robot(
                canvas,
                x,
                y,
                COLORS["robot"],
                pixel_size,
                f"pixel{x}-{y}",
            )


def change_to_void(x, y):  # changement d'un pixel en void
    change_color(x, y, 1, "void")


def change_to_trash(x, y):  # changement d'un pixel en déchet
    change_color(x, y, 1, "trash")
    global number_trash
    gazebo("trash", x, y)
    mTSP()


def change_to_robot(x, y):  # changement d'un pixel en robot
    change_color(x, y, 3, "robot")
    gazebo("warthog", x, y)
    mTSP()


def change_to_drone(x, y):  # changement d'un pixel en drone
    change_color(x, y, 3, "drone")
    gazebo("intelaero", x, y)


def change_to_obstacle(x, y):  # changement d'un pixel en obstacle
    change_color(x, y, 1, "obstacle")
    gazebo("obstacle", x, y)


def change_to_leaf(x, y):  # changement d'un pixel en feuillage
    change_color(x, y, 2, "feuillage")


number_tree = 0


def change_to_tree(x, y):  # changement d'un pixel en arbre (tronc + feuillage)
    radius = 4
    global number_tree
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                new_x, new_y = x + i, y + j
                if 0 <= new_x < len(map[0]) and 0 <= new_y < len(map):
                    change_color(new_x, new_y, 2, "feuillage")

    change_color(x, y, 1, "tronc")
    gazebo("tree", x, y)
    number_tree += 1


###################################################
###boutons fonctions avancées
###################################################


def print_map(map):  # affichage de la matrice dans la console
    color_mapping = {
        "feuillage": Fore.GREEN,
        "tronc": Fore.BLACK,
        "obstacle": Fore.RED,
        "dechet": Fore.BLUE,
        "void": Fore.WHITE,
        "robot": Fore.YELLOW,
        "drone": Fore.MAGENTA,
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


def filter_objects_by_name(objects_list, name):
    return [obj for obj in objects_list if name in obj]


def reset_grid(tout=False):  # réinitialisation de la grille
    for y in range(len(map)):
        for x in range(len(map[y])):
            change_to_void(x, y)
    command = "cd ~/catkin_ws && source devel/setup.bash && rosservice call /gazebo/get_world_properties"  # noqa: E501
    try:
        output = subprocess.check_output(command, shell=True, executable="/bin/bash")
        output = output.decode("utf-8")  # Convert bytes to string
        lines = output.split("\n")  # Split output into lines
        model_names = [
            line.strip()[2:] for line in lines if line.strip().startswith("-")
        ]  # Extract model names
        print(model_names)
    except Exception as e:
        print(f"Failed to launch Gazebo: {e}")
    filtered_objects = filter_objects_by_name(model_names, "tree")
    filtered_objects += filter_objects_by_name(model_names, "trash")
    filtered_objects += filter_objects_by_name(model_names, "warthog")
    filtered_objects += filter_objects_by_name(model_names, "intelaero")
    filtered_objects += filter_objects_by_name(model_names, "obstacle")
    print(filtered_objects)
    for obj in filtered_objects:
        gazebo_delete(obj)


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
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)

canvas.grid(row=3, column=0, columnspan=2)  

canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="white")


drones = []

for i in range(2):
    for j in range(2):
        start_x = i * taille_section
        end_x = start_x + taille_section - 1
        start_y = j * taille_section
        end_y = start_y + taille_section - 1
        drone = Drone_courant(start_x, end_x, start_y, end_y, (2 * i + j + 1), canvas)
        drones.append(drone)

draw_map(canvas, window_size)
draw_drones(canvas)

trash_button = bouton("Déchet au hasard", place_random_trash, "black")
trash_button.grid(row=0, column=0, sticky="w")

tree_button = bouton("Tronc au hasard", place_random_tree, "green")
tree_button.grid(row=1, column=0, sticky="w")

trash_cycle_button = bouton("Déchet fréquents", cycle_trash, "black")
trash_cycle_button.grid(row=2, column=0, sticky="w")

tree_button = bouton("Afficher matrice dans console", lambda: print_map(map), "blue")
tree_button.grid(row=0, column=1, sticky="e")

reset_button = bouton("Réinitialiser environnement", reset_grid, "red")
reset_button.grid(row=1, column=1, sticky="e")
reset2_button = bouton("Réinitialiser tout", reset_grid, "red")
reset2_button.grid(row=2, column=1, sticky="e")


smooth_move_button = bouton("Move", start_drones)
smooth_move_button.grid(row=0, column=0, sticky="e")  

pause_button = bouton("Pause", pause_drones)
pause_button.grid(row=1, column=0,  sticky="e")

stop_button = bouton("Stop and Exit", stop_all_drones_and_exit)
stop_button.grid(row=2, column=0, sticky="e")



root.mainloop()
