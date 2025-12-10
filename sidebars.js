// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'intro/welcome',
        'intro/foundations',
        'intro/hardware-guide',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS2 Fundamentals',
      items: [
        'module1-ros2/overview',
        'module1-ros2/nodes-topics-services',
        'module1-ros2/rclpy-python-bridge',
        'module1-ros2/urdf-humanoids',
        'module1-ros2/project',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin & Simulation',
      items: [
        'module2-digital-twin/gazebo-basics',
        'module2-digital-twin/urdf-sdf',
        'module2-digital-twin/sensors-simulation',
        'module2-digital-twin/unity-visualization',
        'module2-digital-twin/project',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac Sim & vSLAM',
      items: [
        'module3-isaac/isaac-sim',
        'module3-isaac/isaac-ros',
        'module3-isaac/vslam-navigation',
        'module3-isaac/vslam-navigation-completion',
        'module3-isaac/project-summary',
      ],
    },
  ],
};

module.exports = sidebars;