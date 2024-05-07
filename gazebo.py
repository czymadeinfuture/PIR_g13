import subprocess
import tkinter as tk


def function1():
    command = "cd ~/catkin_ws && source devel/setup.bash && roslaunch gazebo_project trash.launch drone_name:=trash_5 x:=5 y:=3"
    subprocess.run(command, shell=True, executable="/bin/bash")


def function2():
    command = "cd ~/catkin_ws && source devel/setup.bash && rosservice call gazebo/delete_model '{model_name: trash_5}'"  # noqa: E501
    subprocess.run(command, shell=True, executable="/bin/bash")


root = tk.Tk()

button1 = tk.Button(root, text="Ajout déchet", command=function1, bg="green")
button1.pack()

button2 = tk.Button(root, text="Suppression déchet", command=function2, bg="red")
button2.pack()

root.mainloop()
