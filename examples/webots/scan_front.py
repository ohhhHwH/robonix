import rclpy
from sensor_msgs.msg import LaserScan
import time
rclpy.init()
n = rclpy.create_node('scan_probe')
box = {}
n.create_subscription(LaserScan, 'scan', lambda m: box.setdefault('m', m), 10)
t = time.time()
while 'm' not in box and time.time() - t < 6:
    rclpy.spin_once(n, timeout_sec=0.3)
m = box.get('m')
if m is None:
    print('NO_SCAN')
else:
    i = round((0.0 - m.angle_min) / m.angle_increment)
    i = max(0, min(i, len(m.ranges) - 1))
    print('front_m=%.3f min_m=%.3f n=%d' % (m.ranges[i], min(m.ranges), len(m.ranges)))
