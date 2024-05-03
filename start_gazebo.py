import subprocess

###fonction qui lance la simu gazebo tout seul (eviter de retaper tout le temps les lignes de terminal)
command = (
    "cd ~/catkin_ws && source devel/setup.bash && roslaunch gazebo_project map.launch"
)
subprocess.run(command, shell=True, executable="/bin/bash")
