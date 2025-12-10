// better-auth/config.ts
import { BetterAuthOptions } from "better-auth";
import { betterAuth } from "better-auth";

// Configuration for user profile collection during registration
const authOptions: BetterAuthOptions = {
  database: {
    url: process.env.NEON_DB_URL!,
    type: "postgres"
  },
  secret: process.env.BETTER_AUTH_SECRET,
  baseURL: process.env.NEXTAUTH_URL || "http://localhost:3000",
  trustHost: true,
  socialProviders: {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }
  },
  // Custom fields for humanoid robot profile data
  user: {
    fields: {
      // Hardware information collected during registration
      hardware_gpu: 'boolean', // RTX 4070+ GPU
      hardware_jetson: 'boolean', // Owns a Jetson kit
      hardware_robot: 'boolean', // Has a real robot
      // Software background
      python_experience: 'integer', // Years with Python
      ros_experience: 'string', // ROS1, ROS2, or Both
      linux_proficiency: 'integer', // Linux proficiency (0-10 scale)
      isaac_rl_experience: 'integer', // Isaac/RL experience (0-10 scale)
      // Primary goal
      primary_goal: 'string', // Learning, Building real humanoid, Research
      skill_level: 'string' // beginner, intermediate, advanced
    }
  },
  // Multi-step signup wizard
  signUp: {
    fields: {
      // Add the custom fields to signup form
      hardware_gpu: {
        type: 'boolean',
        required: false
      },
      hardware_jetson: {
        type: 'boolean',
        required: false
      },
      hardware_robot: {
        type: 'boolean',
        required: false
      },
      python_experience: {
        type: 'number',
        required: false
      },
      ros_experience: {
        type: 'select',
        options: ['none', 'ros1', 'ros2', 'both'],
        required: false
      },
      linux_proficiency: {
        type: 'number',
        required: false
      },
      isaac_rl_experience: {
        type: 'number',
        required: false
      },
      primary_goal: {
        type: 'select',
        options: ['learning', 'building', 'research'],
        required: false
      }
    },
    // Custom multi-step wizard for registration
    customFormFields: [
      {
        id: 'hardware_questions',
        type: 'group',
        fields: [
          {
            id: 'hardware_gpu',
            type: 'checkbox',
            label: 'Do you have an RTX 4070 or higher GPU?',
            description: 'This helps us tailor content for GPU-intensive tasks'
          },
          {
            id: 'hardware_jetson',
            type: 'checkbox',
            label: 'Do you own a Jetson kit?',
            description: 'This helps us provide Jetson-specific content'
          },
          {
            id: 'hardware_robot',
            type: 'checkbox',
            label: 'Do you have a real robot?',
            description: 'This helps us recommend hardware-specific tasks'
          }
        ]
      },
      {
        id: 'software_questions',
        type: 'group',
        fields: [
          {
            id: 'python_experience',
            type: 'range',
            label: 'How many years of experience do you have with Python?',
            min: 0,
            max: 20,
            description: 'Years of Python experience (0-20)'
          },
          {
            id: 'ros_experience',
            type: 'select',
            label: 'What is your experience with ROS?',
            options: [
              { label: 'None', value: 'none' },
              { label: 'ROS1', value: 'ros1' },
              { label: 'ROS2', value: 'ros2' },
              { label: 'Both ROS1 and ROS2', value: 'both' }
            ]
          },
          {
            id: 'linux_proficiency',
            type: 'range',
            label: 'Rate your Linux proficiency (0-10)',
            min: 0,
            max: 10
          },
          {
            id: 'isaac_rl_experience',
            type: 'range',
            label: 'Rate your experience with Isaac Sim or Reinforcement Learning (0-10)',
            min: 0,
            max: 10
          }
        ]
      },
      {
        id: 'goal_section',
        type: 'group',
        fields: [
          {
            id: 'primary_goal',
            type: 'radio-group',
            label: 'What is your primary goal?',
            options: [
              { label: 'Learning', value: 'learning' },
              { label: 'Building a real humanoid', value: 'building' },
              { label: 'Research', value: 'research' }
            ]
          }
        ]
      }
    ]
  },
  // Custom session data
  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
    updateAge: 24 * 60 * 60 // 24 hours
  },
  // Email verification required
  emailVerification: {
    enabled: true,
    sendOnSignUp: true
  }
};

export const auth = betterAuth(authOptions);