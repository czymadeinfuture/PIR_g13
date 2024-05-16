import subprocess

###fonction qui lance la simu gazebo tout seul (eviter de retaper tout le temps les lignes de terminal)
command = "cd ~/PIR_g13 && source ~/catkin_ws/devel/setup.bash && python carte.py"
subprocess.run(command, shell=True, executable="/bin/bash")
