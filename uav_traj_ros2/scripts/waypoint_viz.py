#!/usr/bin/env python3

import rclpy
import math
import numpy as np


from rclpy.node import Node
from geometry_msgs.msg import Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from drone_interfaces.msg import Waypoints
from uav_traj_ros2 import Config

class WaypointViz(Node):
    def __init__(self):
        super().__init__("waypoint_viz")
        self.marker_pub = self.create_publisher(MarkerArray, "waypoint_marker", 10)

        self.wp_sub = self.create_subscription(Waypoints, "/global_waypoints", 
                                               self.waypointCallback, 10)
                
        self.mpc_wp_sub = self.create_subscription(Waypoints, "/mpc_waypoints",
                                                    self.mpcCallback, 10)

        self.counter = 0
        self.id = 0

    def waypointCallback(self,msg)->None:
        """
        Listen for waypoints and publish waypoints
        """
        scale_size = 0.05
        waypoints = msg.points        
        marker_array = MarkerArray()

        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = scale_size
        marker.scale.y = scale_size
        marker.scale.z = scale_size

        marker.color.b = 1.0
        marker.color.a = 1.0    

        for wp in waypoints:

            wp.x = wp.x/Config.SCALE_SIM
            wp.y = wp.y/Config.SCALE_SIM
            wp.z = wp.z/Config.SCALE_SIM
            
            marker.points.append(wp)

            marker.id = self.counter
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "/map"
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()

            marker_array.markers.append(marker)
            self.counter += 1

        self.marker_pub.publish(marker_array)

    def mpcCallback(self,msg)->None:
        pass


def main()->None:
    rclpy.init()
    node = WaypointViz()

    #node.publishWaypointMarkers()
    while rclpy.ok():
        # node.cool_example()
        rclpy.spin_once(node, timeout_sec=0.1)

    rclpy.shutdown()

if __name__=="__main__":
    main()    