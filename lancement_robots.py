import os
import sys
import threading

####faire source ~/catkin_ws/devel/setup.bash avant


sys.path.append(os.path.expanduser("~/catkin_ws/src/pir/packages/gazebo_project/src"))

import example_command


def main():
    # Initialiser le noeud ROS
    example_command.rospy.init_node("listener", anonymous=True)

    # Définir les waypoints pour chaque robot
    waypoints = [
        ["warthog_0", [[10.0, 10.0, 0.0], [-10.0, -10.0, 0.0], [0.0, 0.0, 0.0]]],
        ["warthog_1", [[20.0, 20.0, 0.0], [-20.0, -20.0, 0.0], [0.0, 0.0, 0.0]]],
    ]

    # Créer une instance de Algorithm pour chaque robot et démarrer les threads
    threads = []
    for robot_name, robot_waypoints in waypoints:
        algo = example_command.Algorithm(robot_name, robot_waypoints)
        thread = threading.Thread(target=algo.run)
        thread.start()
        threads.append(thread)

    # Attendre que tous les threads soient terminés
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
