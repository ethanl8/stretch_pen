import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge
from ultralytics import YOLO
from pathlib import Path
import shutil
import subprocess


class OfferPenNode(Node):
    JOINT_NAMES = [
        'joint_gripper_finger_left',
        'joint_lift',
        'joint_arm_l3',
        'joint_arm_l2',
        'joint_arm_l1',
        'joint_arm_l0',
        'joint_wrist_yaw',
        'joint_head_pan',
        'joint_head_tilt',
    ]

    STARTING_RAISE = [
        0.2,
        0.8,
        0.024757600393575246,
        0.024757600393575246,
        0.024757600393575246,
        0.024757600393575246,
        2.1,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    STARTING_OPEN = [
        0.2,
        0.8,
        0.024756104068046757,
        0.024756104068046757,
        0.024756104068046757,
        0.024756104068046757,
        0.25,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    STARTING_POSITION = [
        0.0,
        0.8,
        0.024756104068046757,
        0.024756104068046757,
        0.024756104068046757,
        0.024756104068046757,
        0.25,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    GRIP_POSITION = [
        -0.09,
        0.8,
        0.024759397619537885,
        0.024759397619537885,
        0.024759397619537885,
        0.024759397619537885,
        0.25,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    RESET_POSITION = [
        -0.09,
        0.6,
        0.024757600393575246,
        0.024757600393575246,
        0.024757600393575246,
        0.024757600393575246,
        2.1,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    HOLD_OUT_POSITION = [
        -0.0906052430233963,
        0.86,
        0.02475490700762396,
        0.02475490700762396,
        0.02475490700762396,
        0.02475490700762396,
        2.1,
        0.08059785189388392,
        -0.04062520722347132,
    ]

    def __init__(self):
        super().__init__('offer_pen_node')

        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('person_area_threshold', 750000.0)
        self.declare_parameter('person_height_min_pixels', 300.0)
        self.declare_parameter('person_height_max_pixels', 900.0)
        self.declare_parameter('gripper_lift_min', 0.60)
        self.declare_parameter('gripper_lift_max', 1.00)
        self.declare_parameter(
            'model_path',
            str(Path(__file__).with_name('yolov8n-pose.pt')),
        )


        # Initialize YOLOv8 (using the nano model for fast real-time inference)
        model_path = self.get_parameter('model_path').value
        self.person_area_threshold = self.get_parameter(
            'person_area_threshold'
        ).value
        self.person_height_min_pixels = self.get_parameter(
            'person_height_min_pixels'
        ).value
        self.person_height_max_pixels = self.get_parameter(
            'person_height_max_pixels'
        ).value
        self.gripper_lift_min = self.get_parameter('gripper_lift_min').value
        self.gripper_lift_max = self.get_parameter('gripper_lift_max').value
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Subscribe to the Stretch RealSense camera
        self.sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10,
        )

        # Action client to move the Stretch arm
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory',
        )

        # States: STARTING, GRIPPING, RESETTING, SEARCHING, HOLDING,
        # WAITING, FAULT.
        self.state = 'STARTING'
        self.last_inference_time = 0.0
        self.offer_wait_timer = None
        self.person_in_frame = False
        self.target_gripper_lift = self.HOLD_OUT_POSITION[1]
        self.active_goal_handle = None
        self.startup_timer = self.create_timer(1.0, self.move_to_starting_raise)
        self.starting_pause_timer = None
        self.get_logger().info('Pen Offer Node started.')

        #Sound Settings

        self.declare_parameter(
            'pen_offer_text',
            'Please take a pen.',
        )
        self.declare_parameter('speech_rate', 150)

        self.speech_process = None

        self.tts_program = (
            shutil.which('espeak-ng')
            or shutil.which('espeak')
        )

        if self.tts_program is None:
            self.get_logger().warning(
                "Neither espeak-ng nor espeak was found. "
                "Speech is disabled."
            )
    
    def speak(self, text):
        if self.tts_program is None:
            return

        # Avoid overlapping multiple speech processes
        if (
            self.speech_process is not None
            and self.speech_process.poll() is None
        ):
            return

        rate = str(self.get_parameter('speech_rate').value)

        try:
            self.speech_process = subprocess.Popen(
                [
                    self.tts_program,
                    '-s',
                    rate,
                    '-v',
                    'en-us',
                    text,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            self.get_logger().error(
                f"Could not start speech: {error}"
            )

    def image_callback(self, msg):
        if self.state not in ('SEARCHING', 'HOLDING', 'WAITING'):
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_inference_time < 0.2:
            return
        self.last_inference_time = now

        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 detection (class 0 is 'person')
        results = self.model(cv_image, classes=[0], verbose=False)

        self.person_in_frame = False
        largest_person = None
        for box in results[0].boxes:
            # Calculate bounding box area to estimate distance
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)

            if area > self.person_area_threshold:
                self.person_in_frame = True
                if largest_person is None or area > largest_person[0]:
                    largest_person = (area, y2 - y1)

        if self.state == 'SEARCHING' and self.person_in_frame:
            area, person_height = largest_person
            self.target_gripper_lift = self.lift_for_person_height(person_height)
            self.get_logger().info(
                'Person detected with area %.0f and height %.0f px. '
                'Moving gripper to lift %.2f...'
                % (area, person_height, self.target_gripper_lift)
            )
            self.state = 'HOLDING'
            self.move_to_hold_out()

    def lift_for_person_height(self, person_height):
        height_range = self.person_height_max_pixels - self.person_height_min_pixels
        if height_range <= 0:
            return self.gripper_lift_min

        normalized_height = (
            person_height - self.person_height_min_pixels
        ) / height_range
        normalized_height = max(0.0, min(1.0, normalized_height))
        return self.gripper_lift_min + normalized_height * (
            self.gripper_lift_max - self.gripper_lift_min
        )

    def move_to_starting_raise(self):
        if not self.arm_client.wait_for_server(timeout_sec=0.0):
            return
        self.startup_timer.cancel()
        self.get_logger().info('Moving to starting open raise...')
        self.send_trajectory(
            self.JOINT_NAMES,
            self.STARTING_RAISE,
            4,
            self.move_to_starting_position,
        )


    def move_to_starting_position(self):
        if not self.arm_client.wait_for_server(timeout_sec=0.0):
            return

        self.startup_timer.cancel()
        self.get_logger().info('Moving to starting open position...')
        self.send_trajectory(
            self.JOINT_NAMES,
            self.STARTING_OPEN,
            4,
            self.pause_at_start,
        )

    def pause_at_start(self):
        self.get_logger().info('At starting position. Pausing before closing gripper...')
        self.starting_pause_timer = self.create_timer(
            1.0,
            self.close_gripper_after_pause,
        )

    def close_gripper_after_pause(self):
        self.starting_pause_timer.cancel()
        self.starting_pause_timer = None
        self.close_gripper()

    def close_gripper(self):
        self.state = 'GRIPPING'
        self.get_logger().info('Closing gripper...')
        self.send_trajectory(
            self.JOINT_NAMES,
            self.GRIP_POSITION,
            2,
            self.move_to_reset,
        )

    def move_to_reset(self):
        self.state = 'RESETTING'
        self.get_logger().info('Moving to reset position...')
        self.send_trajectory(
            self.JOINT_NAMES,
            self.RESET_POSITION,
            4,
            self.start_searching,
        )

    def start_searching(self):
        self.state = 'SEARCHING'
        self.person_in_frame = False
        self.get_logger().info('At reset position. Waiting for a person...')

    def move_to_hold_out(self):
        self.get_logger().info('Moving to hold-out position...')
        hold_out_position = list(self.HOLD_OUT_POSITION)
        hold_out_position[1] = self.target_gripper_lift
        self.send_trajectory(
            self.JOINT_NAMES,
            hold_out_position,
            4,
            self.start_offer_wait,
        )

        speech_text = self.get_parameter(
            'pen_offer_text'
        ).value
        self.speak(speech_text)

    def start_offer_wait(self):
        self.state = 'WAITING'
        self.get_logger().info('Waiting 3 seconds for the person to leave...')
        self.offer_wait_timer = self.create_timer(3.0, self.check_person_before_reset)

    def check_person_before_reset(self):
        self.offer_wait_timer.cancel()
        self.offer_wait_timer = None

        if self.person_in_frame:
            self.get_logger().info('Person is still present. Waiting another 3 seconds...')
            self.offer_wait_timer = self.create_timer(3.0, self.check_person_before_reset)
            return

        self.get_logger().info('Person left the frame. Returning to reset position...')
        self.move_to_reset()

    def send_trajectory(self, joint_names, positions, duration, on_complete):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration
        goal.trajectory.points.append(point)

        send_future = self.arm_client.send_goal_async(goal)
        send_future.add_done_callback(
            lambda future: self.trajectory_goal_response(future, on_complete)
        )

    def trajectory_goal_response(self, future, on_complete):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory was rejected')
            self.state = 'FAULT'
            return

        self.active_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.trajectory_result_response(future, on_complete)
        )

    def trajectory_result_response(self, future, on_complete):
        self.active_goal_handle = None
        result = future.result().result
        if result.error_code != 0:
            self.get_logger().error(
                f'Trajectory failed with error code {result.error_code}: '
                f'{result.error_string}'
            )
            self.state = 'FAULT'
            self.get_logger().error(
                'Stopping the offer cycle. Check the robot before restarting.'
            )
            return

        on_complete()

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OfferPenNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    