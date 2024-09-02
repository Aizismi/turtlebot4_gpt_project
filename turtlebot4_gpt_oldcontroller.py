import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from irobot_create_msgs.action import Undock
from openai import OpenAI
import json
import os
import time

class TurtleBot4GPTController(Node):
    def __init__(self):
        super().__init__('turtlebot4_gpt_controller')
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_subscriber = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10)
        self.robot_status = "Ready"
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.undock_action_client = ActionClient(self, Undock, '/undock')

    def status_callback(self, msg):
        self.robot_status = msg.data

    def get_gpt_response(self, user_input):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a robot controller. Translate human instructions into JSON format robot commands. Valid commands are: 'move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop', 'undock'. Include a 'duration' field in seconds for movement commands. The 'undock' command doesn't need a duration."},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.get_logger().error(f'Error communicating with GPT-4: {str(e)}')
            return None

    def interpret_gpt_response(self, gpt_response):
        try:
            command_dict = json.loads(gpt_response)
            return command_dict
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse GPT-4 response as JSON')
            return None

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

    def execute_command(self, command):
        if not isinstance(command, dict):
            self.get_logger().error('Invalid command format')
            return

        if command.get('command') == 'undock':
            success = self.undock_robot()
            if success:
                self.get_logger().info('Robot undocked successfully')
            else:
                self.get_logger().error('Failed to undock robot')
            return

        msg = Twist()
        duration = command.get('duration', 1.0)  # Default duration of 1 second

        if command.get('command') == 'move_forward':
            msg.linear.x = 0.2
        elif command.get('command') == 'move_backward':
            msg.linear.x = -0.2
        elif command.get('command') == 'turn_left':
            msg.angular.z = 0.2
        elif command.get('command') == 'turn_right':
            msg.angular.z = -0.2
        elif command.get('command') == 'stop':
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            self.get_logger().error('Unknown command')
            return

        self.velocity_publisher.publish(msg)
        self.get_logger().info(f'Executing command: {command["command"]} for {duration} seconds')
        
        time.sleep(duration)
        
        # Stop the robot after the duration
        stop_msg = Twist()
        self.velocity_publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = TurtleBot4GPTController()
    
    while True:
        user_input = input("Enter a command for the robot (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        gpt_response = controller.get_gpt_response(user_input)
        if gpt_response:
            print("GPT-4 response:", gpt_response)
            command = controller.interpret_gpt_response(gpt_response)
            if command:
                controller.execute_command(command)
            else:
                print("Failed to interpret GPT-4 response")
        else:
            print("Failed to get response from GPT-4")

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
