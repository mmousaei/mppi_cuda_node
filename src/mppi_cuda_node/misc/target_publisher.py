#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped

def publish_once():
    rospy.init_node('quick_pub', anonymous=True)
    pub = rospy.Publisher('/mppi/target', PoseStamped, latch=True, queue_size=1)
    
    msg = PoseStamped()
    msg.pose.position.x = 9.0
    msg.pose.position.y = 0.0
    msg.pose.position.z = 0.8
    msg.pose.orientation.w = 1.0
    while True:
        msg.pose.position.x += 0.05
        # msg.pose.position.y += 0.05
        pub.publish(msg)
        rospy.sleep(10)

   
    
if __name__ == '__main__':
    publish_once()
