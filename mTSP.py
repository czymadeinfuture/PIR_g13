import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def nearest_neighbor(task_points):
    final_tour = [task_points[0]]
    task_points = np.delete(task_points, 0, axis=0)
    while len(task_points) != 0:
        nrst_nghbr_indx = min(range(len(task_points)), key=lambda x: distance(final_tour[-1], task_points[x]))
        nrst_nghbr = task_points[nrst_nghbr_indx]
        final_tour.append(nrst_nghbr)
        task_points = np.delete(task_points, nrst_nghbr_indx, axis=0)
    final_tour.append(start_point)
    final_tour = np.array(final_tour)
    print(f'Final tour : {final_tour}')
    return final_tour


def distribute_tasks(points, num_robots): #marche bien
    # Divide points equally among robots
    num_points_per_robot = len(points) // num_robots
    print(f'Number of points per robot: {num_points_per_robot}')
    robot_tasks = []

    for i in range(num_robots):
        print(f'Building array for robot {i+1}')
        start_idx = i * num_points_per_robot
        print(f'Start index: {start_idx}')
        end_idx = start_idx + num_points_per_robot if i < num_robots - 1 else len(points)
        print(f'End index: {end_idx}')
        robot_tasks.append(points[start_idx:end_idx])
        print(f'Robot tasks at iteration {i+1} : {robot_tasks}')

    return robot_tasks

def plot_tours(routes,points):
    colors = ['r', 'g', 'b', 'cyan', 'orange']

    plt.figure(figsize=(figure_size,figure_size))

    plt.scatter(points[:, 0], points[:, 1], color ='black', label='Given Points')

    for i, route in enumerate(routes):
        plt.plot(route[:, 0], route[:, 1], color=colors[i], label=f'Route {i+1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Routes')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generate random points for demonstration
    np.random.seed(232)
    num_points = 12
    figure_size = 10
    points = np.random.rand(num_points, 2) * 1000
    

    # Number of robots
    num_robots = 1

    # Distribute tasks among robots
    robot_tasks = distribute_tasks(points, num_robots)

    print(f'Robot tasks : {robot_tasks}')

    # Start point at the center of the grid
    start_point = np.array([5, 5])
    points = np.vstack([start_point, points])

    # Find tours for each robot
    tours = []
    for task in robot_tasks:
        #visited = set()
        print(f'Task : {task}')
        task_with_start = np.vstack([start_point, task])
        print(f'Task with start : {task_with_start}')
        tour = nearest_neighbor(task_with_start)  # Start from the central point
        tours.append(tour)

    # Plot all tours on one graph
    print(f"Tours : {tours}")
    plot_tours(tours, points)
