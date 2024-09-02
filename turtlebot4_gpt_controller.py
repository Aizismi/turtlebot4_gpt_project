import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from irobot_create_msgs.action import Undock, Dock
from openai import OpenAI
import json
import os
import time
import re
import math
import random

try:
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    import cv2
    import numpy as np
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    print("Warning: cv_bridge, OpenCV, or NumPy not available. Some features will be disabled.")
    CV_BRIDGE_AVAILABLE = False

class TurtleBot4CleaningController(Node):
    def __init__(self):
        super().__init__('turtlebot4_cleaning_controller')
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_subscriber = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10)
        if CV_BRIDGE_AVAILABLE:
            self.image_subscription = self.create_subscription(
                Image,
                '/camera/image_raw',
                self.image_callback,
                10)
            self.cv_bridge = CvBridge()
        else:
            self.image_subscription = None
            self.cv_bridge = None
        self.robot_status = "Ready"
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.undock_action_client = ActionClient(self, Undock, '/undock')
        self.dock_action_client = ActionClient(self, Dock, '/dock')
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.gpt4_outputs = []
        self.current_image = None
        self.max_linear_speed = 1.0  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.cleaning_mode = "normal"
        self.battery_level = 100.0  # Assume full battery at start

    def status_callback(self, msg):
        self.robot_status = msg.data
        self.get_logger().info(f"Robot status updated: {self.robot_status}")

    def image_callback(self, msg):
        if CV_BRIDGE_AVAILABLE:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().debug("Received new camera image")
        else:
            self.get_logger().warn("Image processing is not available")

    def get_gpt_response(self, user_input):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                    You are a controller for an advanced domestic TurtleBot4 robot designed for floor cleaning. Translate human instructions into JSON commands to control the robot.

                    Your primary objectives are:
                    1. Ensure the safety of humans, the environment, and the robot at all times.
                    2. Interpret and execute instructions in a way that minimizes any potential harm or damage.
                    3. Maintain ethical behavior and respect the well-being of all parties involved.

                    Available commands:
                    - {"command": "move", "distance": float, "speed": float}
                        - The speed parameter must not exceed 1.0 m/s for safety.
                    - {"command": "rotate", "angle": float, "speed": float}
                        - The speed parameter must not exceed 0.5 rad/s to prevent sudden movements.
                    - {"command": "navigate", "x": float, "y": float, "theta": float}
                        - Navigation commands must ensure the robot does not collide with obstacles or humans.
                    - {"command": "undock"}
                        - Undocking must only be performed when it is safe to do so, and the robot is not in a hazardous position.
                    - {"command": "dock"}
                        - Docking must only be initiated when the target docking station is clear and accessible.
                    - {"command": "stop"}
                        - The stop command must be executed immediately, overriding any other ongoing actions.
                    - {"command": "spiral_clean", "start_radius": float, "end_radius": float, "speed": float}
                        - Spiral cleaning must be performed at a safe speed and radius to avoid collision or damage.
                    - {"command": "wall_follow", "distance": float, "speed": float}
                        - Wall following must maintain a safe distance from the wall and operate at a controlled speed.
                    - {"command": "zig_zag_clean", "width": float, "height": float, "speed": float}
                        - Zig-zag cleaning must be planned and executed to prevent the robot from getting stuck or causing damage.
                    - {"command": "spot_clean", "radius": float, "speed": float}
                        - Spot cleaning must be limited to a safe radius and speed to avoid exceeding the robot's capabilities.
                    - {"command": "random_bounce", "duration": float, "speed": float}
                        - Random bounce cleaning must be performed at a safe speed and for a limited duration to maintain control.
                    - {"command": "edge_clean", "distance": float, "speed": float}
                        - Edge cleaning must be carried out at a safe distance from edges and walls, and at a controlled speed.
                    - {"command": "set_cleaning_mode", "mode": string}
                        - Cleaning modes must be selected based on the specific environment and task, ensuring safety and effectiveness.
                    - {"command": "adjust_brush_roll", "direction": string}
                        - Brush adjustments must be performed cautiously to avoid damage to the robot or surroundings.

                    For complex operations involving camera-based navigation or dirt detection, use Python code wrapped in ```python``` tags.
                    The Python code must include explicit safety checks and ethical considerations to protect humans, the environment, and the robot.

                    Ensure all actions are safe for a domestic environment. Max speed is 1.0 m/s.
                    Provide explanations for your decisions and prioritize safety and ethical behavior above all else.
                    """},
                    {"role": "user", "content": user_input}
                ]
            )
            gpt4_output = response.choices[0].message.content
            self.gpt4_outputs.append({"input": user_input, "output": gpt4_output})
            return gpt4_output
        except Exception as e:
            self.get_logger().error(f'Error communicating with GPT-4: {str(e)}')
            return None

    def execute_gpt_command(self, gpt_response):
        try:
            # Find all JSON blocks in the response
            json_matches = re.findall(r'\{.*?\}', gpt_response, re.DOTALL)
            if json_matches:
                for json_match in json_matches:
                    command = json.loads(json_match)
                    self.execute_json_command(command)
            else:
                # If no JSON blocks are found, check for Python code
                code_match = re.search(r'```python\n(.*?)```', gpt_response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    exec(code, globals())
                    function_name = re.search(r'def (\w+)', code).group(1)
                    execute_command = globals()[function_name]
                    execute_command(self)
                else:
                    # If neither JSON nor Python code is found, log an error
                    self.get_logger().error("No valid command found in GPT-4 response")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in GPT-4 response: {str(e)}")
        except Exception as e:
            self.get_logger().error(f'Error executing GPT-4 command: {str(e)}')


    def execute_json_command(self, command):
        cmd = command.get("command")
        if cmd == "move":
            self.move_robot(command["distance"], command["speed"])
        elif cmd == "rotate":
            self.rotate_robot(command["angle"], command["speed"])
        elif cmd == "navigate":
            self.navigate_to_pose(command["x"], command["y"], command["theta"])
        elif cmd == "undock":
            self.undock_robot()
        elif cmd == "dock":
            self.dock_robot()
        elif cmd == "stop":
            self.stop_robot()
        elif cmd == "spiral_clean":
            self.spiral_clean(command["start_radius"], command["end_radius"], command["speed"])
        elif cmd == "wall_follow":
            self.wall_follow(command["distance"], command["speed"])
        elif cmd == "zig_zag_clean":
            self.zig_zag_clean(command["width"], command["height"], command["speed"])
        elif cmd == "spot_clean":
            self.spot_clean(command["radius"], command["speed"])
        elif cmd == "random_bounce":
            self.random_bounce(command["duration"], command["speed"])
        elif cmd == "edge_clean":
            self.edge_clean(command["distance"], command["speed"])
        elif cmd == "set_cleaning_mode":
            self.set_cleaning_mode(command["mode"])
        elif cmd == "adjust_brush_roll":
            self.adjust_brush_roll(command["direction"])
        else:
            self.get_logger().warn(f"Unknown command: {cmd}")

    def move_robot(self, distance, speed):
        speed = min(abs(speed), self.max_linear_speed)
        msg = Twist()
        msg.linear.x = speed if distance > 0 else -speed
        duration = abs(distance / speed)
        self.velocity_publisher.publish(msg)
        time.sleep(duration)
        self.stop_robot()

    def rotate_robot(self, angle, speed):
        speed = min(abs(speed), self.max_angular_speed)
        msg = Twist()
        msg.angular.z = speed if angle > 0 else -speed
        duration = abs(angle / speed)
        self.velocity_publisher.publish(msg)
        time.sleep(duration)
        self.stop_robot()

    def stop_robot(self):
        msg = Twist()
        self.velocity_publisher.publish(msg)

    def undock_robot(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        future = self.undock_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Undocking goal accepted :)')
            return True
        else:
            self.get_logger().error('Undocking goal rejected :(')
            return False

    def dock_robot(self):
        goal_msg = Dock.Goal()
        self.dock_action_client.wait_for_server()
        future = self.dock_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Docking goal accepted :)')
            return True
        else:
            self.get_logger().error('Docking goal rejected :(')
            return False

    def navigate_to_pose(self, x, y, theta):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = theta
        
        self.nav_to_pose_client.wait_for_server()
        future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Navigation goal accepted :)')
            return True
        else:
            self.get_logger().error('Navigation goal rejected :(')
            return False

    def spiral_clean(self, start_radius, end_radius, speed):
        current_radius = start_radius
        angular_speed = speed / current_radius
        msg = Twist()
        msg.linear.x = speed
        msg.angular.z = angular_speed
        start_time = self.get_clock().now()
        while current_radius <= end_radius:
            self.velocity_publisher.publish(msg)
            elapsed_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
            current_radius = start_radius + (speed * elapsed_time) / (2 * math.pi)
            msg.angular.z = speed / current_radius
        self.stop_robot()

    def wall_follow(self, distance, speed):
        # Placeholder for wall following logic
        self.get_logger().info(f"Wall following for {distance} meters at {speed} m/s")
        self.move_robot(distance, speed)

    def zig_zag_clean(self, width, height, speed):
        num_passes = int(height / (2 * width))
        for i in range(num_passes):
            self.move_robot(width, speed)
            self.rotate_robot(90, speed)
            self.move_robot(2 * width, speed)
            self.rotate_robot(90, speed)
        self.move_robot(width, speed)

    def spot_clean(self, radius, speed):
        for _ in range(4):  # Make 4 concentric circles
            circumference = 2 * math.pi * radius
            self.move_robot(circumference, speed)
            radius -= 0.1  # Decrease radius for next circle

    def random_bounce(self, duration, speed):
        start_time = time.time()
        while time.time() - start_time < duration:
            self.move_robot(1.0, speed)  # Move forward for 1 meter
            self.rotate_robot(random.uniform(-180, 180), speed/2)  # Random turn

    def edge_clean(self, distance, speed):
        # Placeholder for edge cleaning logic
        self.get_logger().info(f"Edge cleaning for {distance} meters at {speed} m/s")
        self.move_robot(distance, speed)

    def set_cleaning_mode(self, mode):
        valid_modes = ["normal", "deep", "quick", "spot"]
        if mode in valid_modes:
            self.cleaning_mode = mode
            self.get_logger().info(f"Cleaning mode set to {mode}")
        else:
            self.get_logger().warn(f"Invalid cleaning mode: {mode}")

    def adjust_brush_roll(self, direction):
        if direction in ["forward", "reverse"]:
            self.get_logger().info(f"Brush roll direction set to {direction}")
        else:
            self.get_logger().warn(f"Invalid brush roll direction: {direction}")

    def detect_dirt(self):
        if CV_BRIDGE_AVAILABLE and self.current_image is not None:
            # Placeholder for dirt detection logic
            return bool(np.random.choice([True, False]))
        else:
            self.get_logger().warn("Dirt detection is not available")
            return False

def main(args=None):
    rclpy.init(args=args)
    controller = TurtleBot4CleaningController()
    
    while True:
        user_input = input("Enter a command for the robot (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        gpt_response = controller.get_gpt_response(user_input)
        if gpt_response:
            print("GPT-4 response:", gpt_response)
            controller.execute_gpt_command(gpt_response)
        else:
            print("Failed to get response from GPT-4")

    # Save GPT-4 outputs to a file
    with open('gpt4_outputs.json', 'w') as f:
        json.dump(controller.gpt4_outputs, f, indent=2)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
