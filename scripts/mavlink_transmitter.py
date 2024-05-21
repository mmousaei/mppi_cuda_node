import time
import math
# Import mavutil
from pymavlink import mavutil
# Imports for attitude
from pymavlink.quaternion import QuaternionBase

class MavlinkTransmitter():

    def __init__(self):
        self.master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
        self.boot_time = time.time()


    def send_attitude_control(self, angular_rates, thrust, quat):
        """ Sets the target attitude while in depth-hold mode.

        'roll', 'pitch', and 'yaw' are angles in degrees.

        """
        self.master.mav.set_attitude_target_send(
            int(1e3 * (time.time() - self.boot_time)), # ms since boot
            self.master.target_system, self.master.target_component,
            # type_mask: 128 means bodyrate and thrust ignore bits are not set
            128,
            # -> attitude quaternion (w, x, y, z | zero-rotation is 1, 0, 0, 0)
            quat,
            angular_rates[0], angular_rates[1], angular_rates[2], thrust # roll rate, pitch rate, yaw rate, thrust
        )

if __name__ == '__main__':

    reader = MavlinkTransmitter()
    # reader.master.wait_heartbeat()
    roll_angle = 0.1
    pitch_angle = 0.2
    yaw_angle = 0.3
    quat = [0.2, 0, 0.2, -0.56]
    print("here")
    while(True):
        reader.set_target_attitude(0, -0.56, quat)
        time.sleep(1) # wait for a second