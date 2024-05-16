#!/usr/bin/env python3
# coding: utf-8

# Auteur : drambeau
# Date   : 12/04/2024


# ##IMPORTATIONS =======================================================
# Import lib standard
import rospy
import math
import numpy as np
import os, time
import tf
import matplotlib.pyplot as plt
from test import *
import sys
import pandas as pd

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
        self.target = rospy.Publisher(
            "/" + self.namespace + "/new_target", Msg_Point, queue_size=10
        )

    # ====================================
    """ 	   Subcriber		"""

    # ====================================
    def __init__subscribers(self):
        rospy.Subscriber(
            "/" + self.namespace + "/groundtruth/odom", Msg_Odometry, self.callback_odom
        )

    # ====================================
    def callback_odom(self, data):
        if not (self.ok):
            # Position initiale dans repère ENU
            self.p0_enu = np.array(
                [
                    data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z,
                ]
            )

            # Orientation initiale repere FLU(t) / ENU
            self.q_enu2flu0 = quaternioninv(
                [
                    data.pose.pose.orientation.x,
                    data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z,
                    data.pose.pose.orientation.w,
                ]
            )
        # Position dans repere ENU
        self.state_penu = np.array(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ]
        )

        self.state_pxy = rotatewithquaternion(self.state_penu, self.q_enu2flu0)[0:2]

        # Orientation repere FLU(t) / ENU
        state_q_flu2enu = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w,
        ]
        self.state_psi = quaternion2psi(
            quaternionproduct(self.q_enu2flu0, state_q_flu2enu)
        )
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

    # ====================================
    """ 	   Subcriber		"""

    # ====================================
    def __init__publishers(self):
        self.target = rospy.Publisher(
            "/" + self.namespace + "/new_target", Msg_Pose, queue_size=10
        )

    # ====================================
    """ 	   Subcriber		"""

    # ====================================
    def __init__subscribers(self):
        rospy.Subscriber(
            "/" + self.namespace + "/groundtruth/odom", Msg_Odometry, self.callback_odom
        )

    # ====================================
    def callback_odom(self, data):
        if not (self.ok):
            # Position initiale dans repère ENU
            self.p0_enu = np.array(
                [
                    data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z,
                ]
            )

            # Orientation initiale repere FLU(t) / ENU
            self.q_enu2flu0 = quaternioninv(
                [
                    data.pose.pose.orientation.x,
                    data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z,
                    data.pose.pose.orientation.w,
                ]
            )
        # Position dans repere ENU
        self.state_penu = np.array(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ]
        )

        self.state_pxy = rotatewithquaternion(self.state_penu, self.q_enu2flu0)[0:2]

        # Orientation repere FLU(t) / ENU
        state_q_flu2enu = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w,
        ]
        self.state_psi = quaternion2psi(
            quaternionproduct(self.q_enu2flu0, state_q_flu2enu)
        )
        yaw, self.pitch, self.roll = quaternion2euler(state_q_flu2enu)

        # Vitesse de rotation
        self.state_omg = data.twist.twist.angular.z
        # OK state
        self.ok = True


# =======================================================
"""		       CLASSES COMMAND		Todo	"""


# =======================================================
class Algorithm(object):
    def __init__(self, robot_name, waypoints=None):
        if "warthog" in robot_name:
            self.warthog = WARTHOG(robot_name)
            if waypoints is None:
                # Utilisez les waypoints par défaut si aucun n'est fourni
                print("pas de waypoints")
            else:
                # Utilisez les waypoints fournis
                self.waypoints_warthog = waypoints

        elif "intelaero" in robot_name:
            self.drone = DRONE(robot_name)

            if waypoints is None:
                # Utilisez les waypoints par défaut si aucun n'est fourni
                print("pas de waypoints")
            else:
                # Utilisez les waypoints fournis
                self.waypoints_drone = waypoints

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
        """Drone waypoints"""
        if hasattr(self, "waypoints_drone"):
            for waypoint in self.waypoints_drone:
                print(waypoint)
                self.Send_drone(waypoint)

        """ Warthog waypoints  """
        if hasattr(self, "waypoints_warthog"):
            for waypoint in self.waypoints_warthog:
                print(waypoint)
                self.Send_warthog(waypoint)

    # ====================================
    """ 	      Run	|Dont touch|	"""

    # ====================================
    def run(self):
        rate = rospy.Rate(100)
        self.exemple()
        while not rospy.is_shutdown():
            rate.sleep()


# ====================================
if __name__ == "__main__":
    rospy.init_node("listener", anonymous=True)

    Al = Algorithm()
    Al.run()
