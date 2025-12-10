// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro/00-welcome',
        'intro/01-foundations',
        'intro/02-hardware-guide',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS2',
      items: [
        'module1-ros2/01-overview',
        'module1-ros2/02-nodes-topics-services',
        'module1-ros2/03-rclpy-python-bridge',
        'module1-ros2/04-urdf-humanoids',
        'module1-ros2/05-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      items: [
        'module2-digital-twin/01-gazebo-basics',
        'module2-digital-twin/02-urdf-sdf',
        'module2-digital-twin/03-sensors-simulation',
        'module2-digital-twin/04-unity-visualization',
        'module2-digital-twin/05-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac',
      items: [
        'module3-isaac/01-isaac-sim',
        'module3-isaac/02-isaac-ros',
        'module3-isaac/03-vslam-navigation',
        'module3-isaac/04-reinforcement-learning',
        'module3-isaac/05-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA',
      items: [
        'module4-vla/01-vision-language-action',
        'module4-vla/02-whisper-voice-commands',
        'module4-vla/03-llm-task-planning',
        'module4-vla/04-capstone-project',
        'module4-vla/05-final-deployment',
      ],
    },
  ],
};

module.exports = sidebars;