#!/usr/bin/env python3
# coding: utf-8

# Auteur : drambeau
# Date   : 12/04/2024


# ##IMPORTATIONS =======================================================
# Import lib standard
import math
import numpy as np
import os, time
import tf
import rospy
import matplotlib.pyplot as plt
from test import *
import sys
import pandas as pd
import random
import time

from math_utils import euler2quaternion
from math_utils import quaternion2euler
from math_utils import euclidian_norm
from math_utils import quaternion2psi
from math_utils import quaternioninv
from math_utils import quaternionproduct
from math_utils import rotatewithquaternion


from geometry_msgs.msg import Point as Msg_Point
from geometry_msgs.msg import Pose as Msg_Pose
from std_msgs.msg import Bool as Msg_Bool

from nav_msgs.msg import Odometry as Msg_Odometry


map_size = 72
#section_size = map_size // 2
block_size= 3
nb = map_size // block_size

 
# =======================================================
"""		     CLASSES WARTHOG			"""
# =======================================================
class WARTHOG(object):
    def __init__(self, namespace):
    
        self.namespace = namespace
        # Pose initiale
        self.p0_enu = None
        self.q_enu2flu0 = None
        # Vecteur d'état
        self.state_penu = None
        self.state_pxy = None
        self.state_psi = None
        self.state_omg = None
        self.roll = None
        self.pitch = None

        self.ok = False
        """| init of the subcriber |"""
        self.__init__subscribers()
        """| init of the publisher |"""
        self.__init__publishers()
        
    # ====================================
    """ 	   Publisher		"""
    # ====================================
    def __init__publishers(self):
        self.target = rospy.Publisher("/"+self.namespace+"/new_target", Msg_Point, queue_size=10)

    # ====================================
    """ 	   Subcriber		"""
    # ====================================		
    def __init__subscribers(self):
        rospy.Subscriber("/"+self.namespace+"/groundtruth/odom", Msg_Odometry, self.callback_odom)
        
    # ====================================
    def callback_odom(self, data):
    
        if not(self.ok):
            # Position initiale dans repère ENU
            self.p0_enu = np.array([data.pose.pose.position.x,
                                    data.pose.pose.position.y,
                                    data.pose.pose.position.z])
            
            
            # Orientation initiale repere FLU(t) / ENU
            self.q_enu2flu0 = quaternioninv([
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w])
        # Position dans repere ENU
        self.state_penu = np.array([
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z])
            
        self.state_pxy = rotatewithquaternion(
            self.state_penu,
            self.q_enu2flu0)[0:2]
            
        # Orientation repere FLU(t) / ENU
        state_q_flu2enu = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w]
        self.state_psi = quaternion2psi(quaternionproduct(self.q_enu2flu0, state_q_flu2enu))
        yaw, self.pitch, self.roll = quaternion2euler(state_q_flu2enu)


        # Vitesse de rotation
        self.state_omg = data.twist.twist.angular.z
        # OK state
        self.ok = True
    
        
# =======================================================
"""		     CLASSES DRONE			"""
# =======================================================  
class DRONE(object):
    def __init__(self, namespace):
        self.namespace = namespace
        # Pose initiale
        self.p0_enu = None
        self.q_enu2flu0 = None
        # Vecteur d'état
        self.state_penu = None
        self.state_pxy = None
        self.state_psi = None
        self.state_omg = None
        self.roll = None
        self.pitch = None
        
        self.ok = False
        """| init of the subcriber |"""
        self.__init__subscribers()
        """| init of the publisher |"""
        self.__init__publishers()

        self.x, self.y =  0 , 0
        self.target_x, self.target_y = self.x, self.y
        
        self.visited = [[0]*nb for _ in range(nb)]
        self.visited_time = [[0]*nb for _ in range(nb)]

        self.last_direction = None 
        self.min_visit = 0

        self.waypoints= [[0.5,0.5,10.0,0.0,0.0,0.0]]

    def nc_drone_ts(self):
        min_visits = float('inf')
        oldest_time = float('inf')
        turn = 0

        best_directions = []
        directions = {
            'up': (0, 3),
            'down': (0, -3),
            'left': (-3, 0),
            'right': (3, 0)
        }

        for direction, (dx, dy) in directions.items():
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < map_size and 0 <= ny < map_size:
                visits = self.visited[ny // block_size][nx // block_size]
                last_time = self.visited_time[ny // block_size][nx // block_size]
                if visits < min_visits:
                    min_visits = visits
                    oldest_time = last_time
                    best_directions = [direction]

                elif visits == min_visits and last_time < oldest_time:
                    oldest_time = last_time
                    best_directions = [direction]

                elif visits == min_visits and last_time == oldest_time:
                    best_directions.append(direction)

        if best_directions:
            if self.last_direction in best_directions:
                best_direction = self.last_direction
            else:
                best_direction = random.choice(best_directions)  

            self.last_direction = best_direction

        if best_direction == "up" and self.y > 0:
            turn = 90.0
            self.target_y += block_size
        elif best_direction == "down" and self.y < map_size - 1:
            self.target_y -= block_size
            turn = 270.0
        elif best_direction == "left" and self.x > 0:
            self.target_x -= block_size
            turn = 180
        elif best_direction == "right" and self.x < map_size - 1:
            self.target_x += block_size
            turn = 0.0
        else:
            return  
        
        self.visited[self.y // block_size][self.x // block_size] += 1
        self.visited_time[self.y // block_size][self.x // block_size]= time.time() 
        self.waypoints.append([(self.target_x+0.5),(self.target_y+0.5),10.0,0.0,0.0,turn])
        self.x, self.y = self.target_x, self.target_y

    def all_covered(self):
        for y in range(nb):
            for x in range(nb):
                if self.visited[y][x] == 0:
                    return False
        return True
    
    def run_drone(self):
        while not self.all_covered():
            self.nc_drone_ts()
            time.sleep(0.1)  

        print("All accessible areas have been covered.")
    
    # ====================================
    """ 	   Subcriber		"""
    # ====================================		
    def __init__publishers(self):
        self.target = rospy.Publisher("/"+self.namespace+"/new_target", Msg_Pose, queue_size=10)

    # ====================================
    """ 	   Subcriber		"""
    # ====================================		
    def __init__subscribers(self):
        rospy.Subscriber("/"+self.namespace+"/groundtruth/odom", Msg_Odometry, self.callback_odom)
        
    # ====================================
    def callback_odom(self, data):
    
        if not(self.ok):
            # Position initiale dans repère ENU
            self.p0_enu = np.array([data.pose.pose.position.x,
                                    data.pose.pose.position.y,
                                    data.pose.pose.position.z])
            
            
            # Orientation initiale repere FLU(t) / ENU
            self.q_enu2flu0 = quaternioninv([
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w])
        # Position dans repere ENU
        self.state_penu = np.array([
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z])
            
        self.state_pxy = rotatewithquaternion(
            self.state_penu,
            self.q_enu2flu0)[0:2]
            
        # Orientation repere FLU(t) / ENU
        state_q_flu2enu = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w]
        self.state_psi = quaternion2psi(quaternionproduct(self.q_enu2flu0, state_q_flu2enu))
        yaw, self.pitch, self.roll = quaternion2euler(state_q_flu2enu)


        # Vitesse de rotation
        self.state_omg = data.twist.twist.angular.z
        # OK state
        self.ok = True
    
        
    
    

# =======================================================
"""		       CLASSES COMMAND		Todo	"""
# =======================================================  
class Algorithm(object):
    def __init__(self):
        self.warthog = WARTHOG("warthog_0")
        self.drone = DRONE("intelaero_0")
        self.drone.run_drone()
        self.waypoints_drone =self.drone.waypoints
        
        # =================================================
        """ waypoint for drones define by the x, y, z, rotx, roty and rotz value  """
        """                     Todo                    """
        # =================================================

        #waypoint for warthog define by the x, y and z value even if the z is already 0                  
        self.waypoints_warthog = [[0.0,  0.0,  0.0],
                                  [10.0, 0.0,  0.0], 
                                  [5.0,  10.0, 0.0]]
    

    # =================================================
    """ Send the waypoint to gazebo   |Dont touch|  """
    # =================================================
    def Send_drone(self, waypoint):
        msg = Msg_Pose()
        msg.position.x = waypoint[0]
        msg.position.y = waypoint[1]
        msg.position.z = waypoint[2]
        
        ori = euler2quaternion(waypoint[5], waypoint[4], waypoint[3])
        msg.orientation.x = ori[0]
        msg.orientation.y = ori[1]
        msg.orientation.z = ori[2]
        msg.orientation.w = ori[3]
        
        print(msg)
        self.drone.target.publish(msg)

    # =================================================    
    """ Send the waypoint to gazebo  |Dont touch|   """
    # =================================================    
    def Send_warthog(self, waypoint):
        msg = Msg_Point()
        msg.x = waypoint[0]
        msg.y = waypoint[1]
        msg.z = waypoint[2]
        self.warthog.target.publish(msg)
        
    # ================================================= 
    """ Take each waypoints for sending to gazebo   """   
    # =================================================

    def exemple(self):
        """ Drone waypoints  """   
        for waypoint in self.waypoints_drone:
            print(waypoint)
            self.Send_drone(waypoint)

        """ Warthog waypoints  """       
        for waypoint in self.waypoints_warthog:
            print(waypoint)
            self.Send_warthog(waypoint)

     # ====================================
    """ 	      Run	|Dont touch|	"""
    # ====================================
    def run(self):
        rate = rospy.Rate(100)
        
        while not rospy.is_shutdown():
            rate.sleep()
            self.exemple()               
                 

            
        
# ====================================
if __name__ == "__main__":
    rospy.init_node('listener', anonymous=True)

    Al = Algorithm()
    Al.run()
    
