#!/bin/bash

# Navigate to the project directory
cd ~/turtlebot4_gpt_project

# Check if virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Activate the virtual environment
source venv/bin/activate

# Install required packages (if not already installed)
pip install rclpy openai numpy pybullet

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi


# Set the OpenAI API key
export OPENAI_API_KEY='sk-proj-I6TXJvSry5F4Ga2LlNTtT3BlbkFJc2b5jBzNQdYAvWFf84JW'

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Check if the API key is valid
echo "Checking API key validity..."
API_CHECK=$(python3 -c "
from openai import OpenAI
import sys

try:
    client = OpenAI(api_key='$OPENAI_API_KEY')
    models = client.models.list()
    print('API key is valid.')
    sys.exit(0)
except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
")

echo $API_CHECK

if [ $? -ne 0 ]; then
    echo "API key validation failed. Please check your API key."
    deactivate
    exit 1
fi

# Run the Python script
python3 turtlebot4_gpt_controller.py

# Deactivate the virtual environment
deactivate
