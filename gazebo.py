import os
import random
import subprocess
import sys
import threading
import time
import tkinter as tk

import subprocess


def function1():
    # os.system(
    #     "cd ~/catkin_ws && catkin build && source devel/setup.bash && roslaunch gazebo_project trash.launch drone_name:=trash_5 x:=5 y:=3"
    # )
    command = "cd ~/catkin_ws && catkin build && source devel/setup.bash && roslaunch gazebo_project trash.launch drone_name:=trash_5 x:=5 y:=3"
    subprocess.run(command, shell=True, executable="/bin/bash")


def function2():
    command = "cd ~/catkin_ws && catkin build && source devel/setup.bash && rosservice call gazebo/delete_model '{model_name: trash_5}'"
    subprocess.run(command, shell=True, executable="/bin/bash")


root = tk.Tk()

button1 = tk.Button(root, text="Button 1", command=function1)
button1.pack()

button2 = tk.Button(root, text="Button 2", command=function2)
button2.pack()

root.mainloop()
