#!/usr/bin/env python3
"""
Project Initialization Script for Physical AI & Humanoid Robotics Textbook
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directories():
    """
    Create all necessary directories for the project
    """
    directories = [
        'docs/intro',
        'docs/module1-ros2',
        'docs/module2-digital-twin', 
        'docs/module3-isaac',
        'docs/module4-vla',
        'src/components',
        'src/theme',
        'src/css',
        'api/routers',
        'better-auth',
        'scripts',
        'static',
        'robot_description',
        'training',
        'training/algorithms',
        'training/environments',
        'training/utils',
        'isaac_env',
        'configs',
        'launch',
        'history/prompts/general',
        'history/prompts/004-physical-ai-book',
        'specs/004-physical-ai-book',
        'specs/004-physical-ai-book/checklists',
        'specs/004-physical-ai-book/contracts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[SUCCESS] Created directory: {directory}")

def create_initial_files():
    """
    Create initial essential files
    """
    # Create package.json if it doesn't exist
    if not os.path.exists('package.json'):
        with open('package.json', 'w') as f:
            f.write("""{
  "name": "physical-ai-book",
  "version": "0.0.1",
  "private": true,
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle",
    "deploy": "docusaurus deploy",
    "clear": "docusaurus clear",
    "serve": "docusaurus serve",
    "write-translations": "docusaurus write-translations",
    "write-heading-ids": "docusaurus write-heading-ids",
    "dev": "docusaurus start --port 3000",
    "index": "node scripts/index-to-qdrant.ts"
  },
  "dependencies": {
    "@docusaurus/core": "^3.1.0",
    "@docusaurus/preset-classic": "^3.1.0",
    "@docusaurus/module-type-aliases": "^3.1.0",
    "@docusaurus/tsconfig": "^3.1.0",
    "@docusaurus/types": "^3.1.0",
    "@giscus/react": "^2.4.0",
    "@mdx-js/react": "^3.0.0",
    "better-auth": "^0.2.1",
    "clsx": "^2.0.0",
    "prism-react-renderer": "^2.3.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "shadcn": "^1.0.0",
    "tailwindcss": "^3.3.0",
    "@headlessui/react": "^1.7.17",
    "@heroicons/react": "^2.0.18",
    "google-generativeai": "^0.1.3",
    "qdrant-js": "^1.6.0"
  },
  "devDependencies": {
    "@docusaurus/module-type-aliases": "^3.1.0",
    "@docusaurus/types": "^3.1.0",
    "@types/react": "^18.2.42",
    "@types/node": "^20.11.17",
    "typescript": "^5.3.3"
  },
  "browserslist": {
    "production": [
      ">0.5%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 3 chrome version",
      "last 3 firefox version",
      "last 5 safari version"
    ]
  },
  "engines": {
    "node": ">=18.0"
  }
}""")
        print("✓ Created package.json")
    
    # Create .env.example if it doesn't exist
    if not os.path.exists('.env.example'):
        with open('.env.example', 'w') as f:
            f.write("""# Environment variables for Physical AI & Humanoid Robotics project

# Gemini API Configuration
GEMINI_API_KEY=your_key_here

# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_api_key_here

# Neon Serverless Postgres Configuration
NEON_DB_URL=postgresql://user:password@ep-xxx.us-east-1.aws.neon.tech/dbname

# Better-Auth Configuration
BETTER_AUTH_SECRET=your_secret_here

# Isaac Sim Configuration
ISAAC_SIM_PATH=/path/to/isaac/sim
""")
        print("✓ Created .env.example")
    
    # Create docusaurus.config.ts if it doesn't exist
    if not os.path.exists('docusaurus.config.ts'):
        with open('docusaurus.config.ts', 'w') as f:
            f.write("""import { themes as prismThemes } from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'An Interactive Textbook with Personalised RAG Chatbot',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'your-organization',
  projectName: 'physical-ai-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disable blog for textbook
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            type: 'localeDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/your-organization/physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro/00-welcome',
              },
              {
                label: 'ROS2 Fundamentals',
                to: '/docs/module1-ros2/01-overview',
              },
              {
                label: 'Digital Twin',
                to: '/docs/module2-digital-twin/01-gazebo-basics',
              },
              {
                label: 'Isaac Sim & vSLAM',
                to: '/docs/module3-isaac/01-isaac-sim-basics',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-organization/physical-ai-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'yaml', 'json', 'typescript', 'protobuf'],
      },
    }),
  
  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {
            to: '/docs/intro/00-welcome', 
            from: ['/docs/Intro', '/docs/welcome'],
          },
        ],
      },
    ],
  ],
};

export default config;
""")
        print("✓ Created docusaurus.config.ts")
    
    # Create sidebars.js if it doesn't exist
    if not os.path.exists('sidebars.js'):
        with open('sidebars.js', 'w') as f:
            f.write("""// @ts-check

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
      label: 'Module 1: ROS2 Fundamentals',
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
      label: 'Module 2: Digital Twin & Simulation',
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
      label: 'Module 3: Isaac Sim & vSLAM',
      items: [
        'module3-isaac/01-isaac-sim-basics',
        'module3-isaac/02-isaac-ros-integration',
        'module3-isaac/03-vslam-navigation',
        'module3-isaac/04-vslam-navigation-completion',
        'module3-isaac/05-project-summary',
      ],
    },
  ],
};

module.exports = sidebars;
""")
        print("✓ Created sidebars.js")
    
    # Create tsconfig.json if it doesn't exist
    if not os.path.exists('tsconfig.json'):
        with open('tsconfig.json', 'w') as f:
            f.write("""{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "allowJs": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true
  },
  "ts-node": {
    "esm": true,
    "experimentalSpecifierResolution": "node"
  },
  "include": ["src", "docs", "scripts", "api", "docusaurus.config.ts", "sidebars.js"],
  "exclude": ["node_modules", "build", "dist"]
}""")
        print("✓ Created tsconfig.json")

def create_sample_content():
    """
    Create a sample welcome page to verify setup
    """
    sample_content = """---
sidebar_position: 1
---

# Welcome to Physical AI & Humanoid Robotics

## Introduction

Welcome to the comprehensive interactive textbook on Physical AI and Humanoid Robotics! This textbook covers everything from basic concepts to advanced reinforcement learning techniques for humanoid robots.

### What You'll Learn

- ROS2 fundamentals for humanoid control
- Digital twin development with Isaac Sim
- Visual SLAM for navigation
- Reinforcement learning for complex behaviors
- Sensor integration and perception
- Advanced control algorithms

### Getting Started

To get the most out of this textbook, we recommend following the modules in order:
1. Start with the ROS2 fundamentals
2. Move to digital twin concepts
3. Explore Isaac Sim and vSLAM
4. Implement reinforcement learning algorithms

### Personalization

Use the personalization button to adjust content complexity based on your experience level.

### Multilingual Support

Use the translation button to read this content in Urdu.
"""
    
    with open('docs/intro/00-welcome.mdx', 'w') as f:
        f.write(sample_content)
    
    print("✓ Created sample content file: docs/intro/00-welcome.mdx")

def setup_git():
    """
    Initialize git if not already initialized
    """
    if not os.path.exists('.git'):
        try:
            subprocess.run(['git', 'init'], check=True, capture_output=True)
            print("✓ Initialized git repository")
        except subprocess.CalledProcessError:
            print("⚠ Could not initialize git repository (git not available)")

def main():
    """
    Main execution function for project initialization
    """
    print("Initializing Physical AI & Humanoid Robotics Project...")
    print("="*60)
    
    # Create project directories
    print("Creating directories...")
    create_directories()
    
    # Create initial files
    print("\nCreating initial configuration files...")
    create_initial_files()
    
    # Create sample content
    print("\nCreating sample content...")
    create_sample_content()
    
    # Initialize git
    print("\nSetting up git...")
    setup_git()
    
    # Instructions for user
    print("\n" + "="*60)
    print("🎉 Project initialization complete!")
    print("\nNext steps:")
    print("1. Fill in your environment variables in a .env file (copy from .env.example)")
    print("2. Install dependencies: npm install")
    print("3. Start development server: npm run dev")
    print("4. Add your content to the docs/ directory")
    print("5. Customize components in src/components/")
    print("\nThe project structure is now ready for development.")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Physical AI & Humanoid Robotics project is ready for development!")
    else:
        print("\n❌ Project initialization failed")
        sys.exit(1)
