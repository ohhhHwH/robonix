import os, time, rclpy
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
print('RMW=', os.environ.get('RMW_IMPLEMENTATION'), 'DOMAIN=', os.environ.get('ROS_DOMAIN_ID'))
rclpy.init()
n = rclpy.create_node('scan_probe4')
t0 = time.time()
names = {}
while time.time() - t0 < 4:
    names = dict(n.get_topic_names_and_types())
    if any('scan' in k for k in names):
        break
    rclpy.spin_once(n, timeout_sec=0.2)
print('scan_topics=', {k: v for k, v in names.items() if 'scan' in k})
got = {}
q = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
n.create_subscription(LaserScan, '/scanner/scan', lambda m: got.setdefault('m', m), q)
t = time.time()
while 'm' not in got and time.time() - t < 10:
    rclpy.spin_once(n, timeout_sec=0.3)
m = got.get('m')
if m is None:
    print('NO_SCAN')
else:
    i = int(round((0.0 - m.angle_min) / m.angle_increment))
    i = max(0, min(i, len(m.ranges) - 1))
    print('front_m=%.3f' % m.ranges[i])
