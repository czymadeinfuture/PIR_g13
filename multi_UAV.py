import tkinter as tk
import random
dt = 1
taille_carte = 10  
canvas_width = 700
canvas_height = 700
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
    def __init__(self,start_x, end_x, start_y, end_y):
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y


        self.scan_radius = 30
        self.drone_size =  pixel_size * 0.2 
        self.x, self.y = self.random_position()
        self.target_x, self.target_y = self.x, self.y
        self.moving = False
        self.path = []
        self.visited = [[0]*taille_carte for _ in range(taille_carte)]
        self.trash_found = []
        self.map_copy= [[0] * taille_carte for _ in range(taille_carte)]  
        self.init_map_copy()
        self.last_direction = None 

        

    def init_map_copy(self):
        for y in range(self.start_y, self.end_y + 1):
            for x in range(self.start_x, self.end_x + 1):
                if map_data[y][x] in (1, 2):  
                    self.map_copy[y][x] = map_data[y][x]

    def all_covered(self):
        for y in range(self.start_y, self.end_y + 1):
            for x in range(self.start_x, self.end_x + 1):
                if self.map_copy[y][x] == 0 and self.visited[y][x] == 0:
                    return False
        return True

    def random_position(self):
        x = random.randint(self.start_x, self.end_x)
        y = random.randint(self.start_y, self.end_y)
        return x, y

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
        start_angle = self.get_scan_start_angle()
        canvas.create_arc(
            center_x - self.scan_radius, center_y - self.scan_radius,
            center_x + self.scan_radius, center_y + self.scan_radius,
            start=start_angle, extent=180, style=tk.CHORD, outline=COLORS["drone"], width=2, tags=tag
        )

    def draw_path(self, canvas):
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                start_x, start_y = self.path[i - 1]
                end_x, end_y = self.path[i]
                canvas.create_line(start_x, start_y, end_x, end_y, fill="#0000FF", width=2, tags="path")

    def get_scan_start_angle(self):
        direction_angle = {
            'up': 0, 
            'down': 180, 
            'left': 90, 
            'right': 270,  
        }
        return direction_angle.get(self.last_direction, 0)  

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


    def start_move(self):
        if not self.moving:
            min_visits = float('inf')
            best_directions = []
            directions = {
                'up': (0, -1),
                'down': (0, 1),
                'left': (-1, 0),
                'right': (1, 0)
            }

            for direction, (dx, dy) in directions.items():
                nx, ny = self.x + dx, self.y + dy
                if self.start_x <= nx <= self.end_x and self.start_y <= ny <= self.end_y:
                        if self.map_copy[ny][nx] == 0: 
                            visits = self.visited[ny][nx]
                            if visits < min_visits:
                                min_visits = visits
                                best_directions = [direction]
                            elif visits == min_visits:
                                best_directions.append(direction)

            if best_directions:
                if self.last_direction in best_directions:
                    best_direction = self.last_direction
                else:
                    best_direction = random.choice(best_directions)  

                self.last_direction = best_direction

            if best_direction == "up" and self.y > 0:
                self.target_y -= 1
            elif best_direction == "down" and self.y < taille_carte - 1:
                self.target_y += 1
            elif best_direction == "left" and self.x > 0:
                self.target_x -= 1
            elif best_direction == "right" and self.x < taille_carte - 1:
                self.target_x += 1
            else:
                return  # Invalid move
            self.vx = 0.10 if self.x < self.target_x else -0.10 if self.x > self.target_x else 0
            self.vy = 0.10 if self.y < self.target_y else -0.10 if self.y > self.target_y else 0
            self.moving = True
            self.visited[self.y][self.x] += 1

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

            if not self.all_covered():
                self.draw_drone(canvas)
                root.after(10, lambda: self.move_to_target(canvas))
        
        

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

    return new_map,map_data

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

def auto_move():
    all_covered = True
    for drone in drones:
        if not drone.all_covered():
            all_covered = False
            if not drone.moving:
                drone.start_move()
                drone.move_to_target(canvas)
    if not all_covered:
        root.after(200, auto_move)
    else:
        print("All accessible areas have been covered.")



root = tk.Tk()
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()


map,map_data = init_map(taille_carte)
drones = []
section_size = taille_carte // 2
for i in range(2):
    for j in range(2):
        start_x = i * section_size
        end_x = start_x + section_size - 1
        start_y = j * section_size
        end_y = start_y + section_size - 1
        drone = Drone(start_x, end_x, start_y, end_y)
        drones.append(drone)


draw_map(canvas) 

draw_drones(canvas)

path_button = tk.Button(root, text="show path", command=toggle_path)
path_button.pack()

smooth_move_button = tk.Button(root, text="move", command=auto_move)
smooth_move_button.pack()
root.mainloop()