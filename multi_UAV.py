import tkinter as tk
import random
import time
import threading

dt = 1
taille_carte = 18
taille_section = taille_carte // 2
taille_bloc = 3
nb = taille_section//taille_bloc
canvas_width = 900
canvas_height = 900
pixel_size = canvas_width // taille_carte 

COLORS = {
    "void": "#FFFFFF",
    "obstacle": "#FF0000",
    "tronc": "#8B4513",
    "feuillage": "#00B500",
    "trash": "#888888",
    "drone": "#FFD700"
}

class Cellule:
    def __init__(self, etage1="void", etage2="void"):
        self.etage1 = etage1
        self.etage2 = etage2

class Drone:
    def __init__(self,start_x, end_x, start_y, end_y,canvas):

        self.list = self.init_list(start_x,start_y)
        self.waypoints= []

        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y

        self.canvas = canvas

        self.scan_radius = 60
        self.drone_size =  pixel_size * 0.2 
        self.x, self.y = self.init_position()

        self.target_x, self.target_y = self.x, self.y
        self.moving = False

        self.path = []

        self.visited = [[0]*nb for _ in range(nb)]
        self.trash_found = []
        self.map_copy= [[0] * taille_carte for _ in range(taille_carte)]  
        self.init_map_copy()

        self.visited_time = [[0]*taille_carte for _ in range(taille_carte)]
        self.last_direction = None 

        self.min_visit = 0

        self.should_run = True

        
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
        canvas.create_oval(
            center_x - self.drone_size, center_y - self.drone_size,
            center_x + self.drone_size, center_y + self.drone_size,
            fill="#FFD700", outline="#000", width=2, tags=tag
        )
        # dessine de zone scanne
        canvas.create_oval(
            center_x - self.scan_radius, center_y - self.scan_radius,
            center_x + self.scan_radius, center_y + self.scan_radius,
            outline="#FFD700", width=2, tags=tag
        )

    def draw_path(self, canvas):
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                start_x, start_y = self.path[i - 1]
                end_x, end_y = self.path[i]
                canvas.create_line(start_x, start_y, end_x, end_y, fill="#0000FF", width=2, tags="path")

    def scan_for_trash(self, canvas):
        radius_pixels = self.scan_radius  
        center_x = (self.x + 0.5) * pixel_size
        center_y = (self.y + 0.5) * pixel_size

        for dx in range(-radius_pixels, radius_pixels + 1):
            for dy in range(-radius_pixels, radius_pixels + 1):
                if dx**2 + dy**2 < radius_pixels**2 and dx**2 + dy**2 > self.drone_size**2:  
                    pixel_x = center_x + dx
                    pixel_y = center_y + dy
                    grid_x = int(pixel_x // pixel_size)
                    grid_y = int(pixel_y // pixel_size)

                    if 0 <= grid_x < taille_carte and 0 <= grid_y < taille_carte:
                        item_id = canvas.find_closest(pixel_x, pixel_y)
                        item_color = canvas.itemcget(item_id, "fill")
                        if self.map_copy[grid_y][grid_x] == 0 and item_color == COLORS["trash"]:
                            #if item_color not in [COLORS["void"], COLORS["tronc"], COLORS["obstacle"], COLORS["feuillage"],COLORS["drone"]]:
                            if (grid_x, grid_y) not in self.trash_found:  
                                self.trash_found.append((grid_x, grid_y))
                                print(f"Trash found at position: ({grid_x+1}, {grid_y+1})")

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
    
    def run(self):
        self.initial_scan(self.canvas)
        while self.should_run:
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

    def stop(self):
        self.should_run = False

    def update_gui(self):
        self.canvas.after(0, self.draw_drone, self.canvas)
        
        

def init_map(num):
    new_map = [[Cellule() for _ in range(num)] for _ in range(num)]
    map_data = [[0]*num for _ in range(num)]
    place_fixed_trees(new_map,map_data,3, 3)  
    new_map[2][7] = Cellule("obstacle") 
    map_data[2][7] = 2

    for _ in range(10):  
        trash_x = random.randint(0, num - 1)
        trash_y = random.randint(0, num - 1)
        if map_data[trash_y][trash_x] == 0:  
            map_data[trash_y][trash_x] = 3 
    
    block_counts = [[0] * (taille_carte//taille_bloc) for _ in range(taille_carte//taille_bloc)]
    for y in range(taille_carte//taille_bloc):
        for x in range(taille_carte//taille_bloc):
            count = sum(1 for dy in range(taille_bloc)
                        for dx in range(taille_bloc)
                        if map_data[y * taille_bloc + dy][x * taille_bloc + dx] in {1, 2})
            block_counts[y][x] = count

    return new_map,map_data,block_counts

    

def place_fixed_trees(map, map_data,x, y):
    radius = 2
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                nx, ny = x + i, y + j
                if 0 <= nx < taille_carte and 0 <= ny < taille_carte:
                    map[ny][nx].etage2 = "feuillage"
                    map_data[ny][nx] = 1
    map[y][x].etage1 = "tronc"  

def draw_pixel(canvas, x, y, cell_type, pixel_size):
    color_dict = {
        0: COLORS["void"], 
        1: COLORS["feuillage"], 
        2: COLORS["obstacle"], 
        3: COLORS["trash"]  
    }
    color = color_dict.get(cell_type, COLORS["void"])

    canvas.create_rectangle(
        x * pixel_size,
        y * pixel_size,
        (x + 1) * pixel_size,
        (y + 1) * pixel_size,
        fill=color,
        outline="black"
    )

def draw_map(canvas):
    pixel_size = canvas_width // len(map)
    for y, row in enumerate(map):
        for x, cell in enumerate(row):
            draw_pixel(canvas, x, y, map_data[y][x], pixel_size)
            #color = COLORS[cell.etage1]
            #draw_pixel(canvas, x, y, color, pixel_size)
            #if cell.etage2 == "feuillage":
                #color = COLORS[cell.etage2]
                #draw_pixel(canvas, x, y, color, pixel_size)

def draw_drones(canvas):
    for drone in drones:
        drone.draw_drone(canvas)

def toggle_path():
    # Function to toggle the display of the path
    for drone in drones:
        drone.draw_path(canvas)


def start_drones():
    for drone in drones:
        if not drone.moving:
            drone_thread = threading.Thread(target=drone.run)
            drone_thread.start()

def stop_all_drones_and_exit():
    for drone in drones:
        drone.stop() 
    root.destroy() 

root = tk.Tk()
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()


map,map_data,block_counts = init_map(taille_carte)
drones = []

for i in range(2):
    for j in range(2):
        start_x = i * taille_section
        end_x = start_x + taille_section - 1
        start_y = j * taille_section
        end_y = start_y + taille_section - 1
        drone = Drone(start_x, end_x, start_y, end_y,canvas)
        drones.append(drone)



draw_map(canvas) 
draw_drones(canvas)

path_button = tk.Button(root, text="show path", command=toggle_path)
path_button.pack()

smooth_move_button = tk.Button(root, text="move", command=start_drones)
smooth_move_button.pack()

stop_button = tk.Button(root, text="Stop and Exit", command=stop_all_drones_and_exit)
stop_button.pack()
root.mainloop()
