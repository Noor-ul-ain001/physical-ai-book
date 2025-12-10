import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

// Import icons
import { 
  FiPlayCircle, 
  FiBookOpen, 
  FiCpu, 
  FiDollarSign,
  FiGitBranch,
  FiUsers,
  FiTrendingUp,
  FiShield,
  FiGlobe,
  FiCode,
  FiLayers,
  FiServer,
  FiCloud,
  FiHardDrive,
  FiWifi,
  FiActivity,
  FiCheckCircle
} from 'react-icons/fi';
import { 
  RiRobot2Line, 
  RiBrainLine, 
  RiHandHeartLine,
  RiLightbulbFlashLine,
  RiCloudLine,
  RiServerLine,
  RiCpuLine
} from 'react-icons/ri';
import { 
  SiNvidia,
  SiRos,
  SiUnity,
  SiUbuntu
} from 'react-icons/si';

// Statistics data
const statsData = [
  { label: 'Learning Hours', value: '120+', icon: <FiBookOpen />, color: '#3B82F6' },
  { label: 'Modules', value: '4', icon: <RiBrainLine />, color: '#8B5CF6' },
  { label: 'Weekly Sessions', value: '13', icon: <FiActivity />, color: '#10B981' },
  { label: 'Capstone Projects', value: '5+', icon: <FiCheckCircle />, color: '#F59E0B' },
];

// Feature cards data
const featuresData = [
  {
    icon: <SiRos />,
    title: 'Robotic Nervous System',
    description: 'Master ROS 2 for robot control with Python agents and URDF',
    color: '#3B82F6'
  },
  {
    icon: <SiUnity />,
    title: 'Digital Twin',
    description: 'Physics simulation with Gazebo & high-fidelity Unity rendering',
    color: '#8B5CF6'
  },
  {
    icon: <SiNvidia />,
    title: 'AI-Robot Brain',
    description: 'NVIDIA Isaac for perception, VSLAM, and synthetic data generation',
    color: '#76B900'
  },
  {
    icon: <RiBrainLine />,
    title: 'Vision-Language-Action',
    description: 'LLMs + Robotics: Voice commands to autonomous actions',
    color: '#EF4444'
  },
  {
    icon: <FiCloud />,
    title: 'Sim-to-Real Transfer',
    description: 'Deploy simulation-trained models to physical robots',
    color: '#10B981'
  },
  {
    icon: <RiServerLine />,
    title: 'Edge Computing',
    description: 'Jetson Orin for real-time inference on physical robots',
    color: '#F59E0B'
  },
];

// Hardware tiers data
const hardwareTiers = [
  {
    tier: 'Digital Twin Workstation',
    description: 'Required per student for simulation and training',
    essential: true,
    components: [
      { name: 'GPU', spec: 'NVIDIA RTX 4070 Ti (12GB) or higher', note: 'Minimum for Isaac Sim', icon: <SiNvidia /> },
      { name: 'CPU', spec: 'Intel i7 13th Gen+ or AMD Ryzen 9', note: 'Physics calculations', icon: <RiCpuLine /> },
      { name: 'RAM', spec: '64GB DDR5', note: '32GB minimum', icon: <FiHardDrive /> },
      { name: 'OS', spec: 'Ubuntu 22.04 LTS', note: 'Required for ROS 2', icon: <SiUbuntu /> }
    ]
  },
  {
    tier: 'Physical AI Edge Kit',
    description: 'For deploying AI to physical systems',
    essential: true,
    components: [
      { name: 'Brain', spec: 'NVIDIA Jetson Orin Nano (8GB)', note: '40 TOPS for edge AI', icon: <RiBrainLine /> },
      { name: 'Vision', spec: 'Intel RealSense D435i', note: 'RGB-D + IMU', icon: <FiActivity /> },
      { name: 'Voice', spec: 'ReSpeaker Mic Array', note: 'Voice command interface', icon: <FiWifi /> }
    ]
  },
  {
    tier: 'Robot Options',
    description: 'Choose based on budget and learning goals',
    essential: false,
    options: [
      { name: 'Budget', robot: 'Hiwonder TonyPi Pro', price: '$600', note: 'Raspberry Pi based' },
      { name: 'Recommended', robot: 'Unitree Go2 Edu', price: '$1,800-3,000', note: 'Quadruped proxy' },
      { name: 'Premium', robot: 'Unitree G1', price: '$16,000', note: 'Advanced humanoid' }
    ]
  }
];

const HomepageHero = () => {
  const { siteConfig } = useDocusaurusContext();

  return (
    <header className={clsx(styles.heroBanner)} style={{ background: 'linear-gradient(135deg, #0F172A 0%, #1E293B 100%)' }}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroText}>
            <div className={styles.badge} style={{ background: '#8B5CF6', color: 'white' }}>🚀 Capstone Quarter</div>
            <h1 className={styles.heroTitle}>
              <span className={styles.titleGradient} style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)', WebkitBackgroundClip: 'text' }}>
                Physical AI
              </span>
              <br />
              & Humanoid Robotics
            </h1>
            <p className={styles.heroSubtitle} style={{ color: '#CBD5E1', maxWidth: '800px' }}>
              Focus on AI Systems in the Physical World. Master embodied intelligence by bridging digital brains 
              with physical bodies. Apply AI to control humanoid robots in simulated and real-world environments.
            </p>
            
            <div className={styles.heroButtons}>
              <Link
                className={clsx(styles.primaryButton, styles.button)}
                to="/docs/intro"
                style={{ background: '#3B82F6', borderColor: '#3B82F6' }}
              >
                <FiPlayCircle className={styles.buttonIcon} />
                Start Learning - 5min ⏱️
              </Link>
              <Link
                className={clsx(styles.secondaryButton, styles.button)}
                to="/docs/quick-start"
                style={{ background: '#1E293B', borderColor: '#475569' }}
              >
                <FiCode className={styles.buttonIcon} />
                View Projects
              </Link>
              <button className={clsx(styles.outlineButton, styles.button)} style={{ borderColor: '#8B5CF6', color: '#8B5CF6' }}>
                <FiGitBranch className={styles.buttonIcon} />
                Join Community
              </button>
            </div>

            {/* Stats Section */}
            <div className={styles.heroStats}>
              {statsData.map((stat, index) => (
                <div key={index} className={styles.statItem} style={{ background: '#1E293B', borderColor: '#334155' }}>
                  <div 
                    className={styles.statIcon}
                    style={{ background: `linear-gradient(135deg, ${stat.color}, ${stat.color}80)` }}
                  >
                    {stat.icon}
                  </div>
                  <div className={styles.statContent}>
                    <h3 className={styles.statValue} style={{ color: 'white' }}>{stat.value}</h3>
                    <p className={styles.statLabel} style={{ color: '#94A3B8' }}>{stat.label}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Hero Visual Section */}
          <div className={styles.heroVisual}>
            {/* Architecture Diagram */}
            <div className={styles.modelContainer} style={{ background: '#1E293B', borderColor: '#334155' }}>
              <div className={styles.modelHeader} style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)' }}>
                <RiRobot2Line className={styles.modelIcon} />
                <span>Course Architecture</span>
              </div>
              <div className={styles.modelPlaceholder}>
                <div className={styles.architectureGrid}>
                  <div className={styles.archItem} style={{ background: '#3B82F620', borderColor: '#3B82F640' }}>
                    <div className={styles.archIcon}>🤖</div>
                    <span>Physical AI</span>
                  </div>
                  <div className={styles.archArrow}>→</div>
                  <div className={styles.archItem} style={{ background: '#8B5CF620', borderColor: '#8B5CF640' }}>
                    <div className={styles.archIcon}>🧠</div>
                    <span>Embodied Intelligence</span>
                  </div>
                  <div className={styles.archArrow}>→</div>
                  <div className={styles.archItem} style={{ background: '#10B98120', borderColor: '#10B98140' }}>
                    <div className={styles.archIcon}>⚡</div>
                    <span>Capstone Project</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quarter Overview */}
            <div className={styles.videoContainer} style={{ background: '#1E293B', borderColor: '#334155' }}>
              <div className={styles.videoHeader} style={{ background: 'linear-gradient(135deg, #10B981, #059669)' }}>
                <FiBookOpen className={styles.videoIcon} />
                <span>Quarter Overview</span>
              </div>
              <div className={styles.videoPlaceholder}>
                <p style={{ color: '#CBD5E1', lineHeight: '1.6' }}>
                  The future of AI extends beyond digital spaces into the physical world. 
                  This capstone quarter introduces Physical AI—AI systems that function in 
                  reality and comprehend physical laws.
                </p>
                <div style={{ marginTop: '15px', color: '#8B5CF6', fontWeight: '600' }}>
                  <FiCheckCircle style={{ marginRight: '8px' }} />
                  Design, simulate, and deploy humanoid robots
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className={styles.featuresGrid}>
          {featuresData.map((feature, index) => (
            <div key={index} className={styles.featureCard} style={{ 
              background: 'linear-gradient(135deg, #1E293B, #0F172A)',
              border: `1px solid ${feature.color}40`,
              boxShadow: `0 4px 20px ${feature.color}20`
            }}>
              <div 
                className={styles.featureIcon}
                style={{ 
                  background: `linear-gradient(135deg, ${feature.color}, ${feature.color}80)`,
                  color: 'white'
                }}
              >
                {feature.icon}
              </div>
              <h3 className={styles.featureTitle} style={{ color: 'white' }}>{feature.title}</h3>
              <p className={styles.featureDescription} style={{ color: '#94A3B8' }}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
};

const HomepageFeatures = () => {
  const [activeSection, setActiveSection] = useState('modules');

  // Modules data
  const modulesData = [
    {
      title: "Module 1: The Robotic Nervous System",
      subtitle: "ROS 2 Fundamentals",
      topics: [
        "ROS 2 Nodes, Topics, and Services",
        "Bridging Python Agents to ROS controllers using rclpy",
        "URDF for humanoid robot description",
        "Weeks 3-5: ROS 2 package development"
      ],
      icon: <SiRos />,
      color: '#3B82F6'
    },
    {
      title: "Module 2: Digital Twin",
      subtitle: "Gazebo & Unity Simulation",
      topics: [
        "Physics simulation and environment building",
        "High-fidelity rendering in Unity",
        "Sensor simulation: LiDAR, Depth Cameras, IMU",
        "Weeks 6-7: Simulation implementation"
      ],
      icon: <SiUnity />,
      color: '#8B5CF6'
    },
    {
      title: "Module 3: AI-Robot Brain",
      subtitle: "NVIDIA Isaac Platform",
      topics: [
        "Isaac Sim: Photorealistic simulation",
        "Isaac ROS: Hardware-accelerated VSLAM",
        "Nav2: Bipedal path planning",
        "Weeks 8-10: Perception pipeline"
      ],
      icon: <SiNvidia />,
      color: '#76B900'
    },
    {
      title: "Module 4: Vision-Language-Action",
      subtitle: "LLMs + Robotics Integration",
      topics: [
        "OpenAI Whisper for voice commands",
        "LLMs for natural language to ROS actions",
        "Capstone: Autonomous Humanoid Project",
        "Weeks 11-13: Conversational robotics"
      ],
      icon: <RiBrainLine />,
      color: '#EF4444'
    }
  ];

  return (
    <section className={styles.mainContent} style={{ background: '#0F172A' }}>
      <div className="container">
        {/* Navigation Tabs */}
        <div className={styles.contentTabs} style={{ background: '#1E293B', borderColor: '#334155' }}>
          <button
            className={clsx(styles.tab, activeSection === 'modules' && styles.activeTab)}
            onClick={() => setActiveSection('modules')}
            style={activeSection === 'modules' ? { 
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              color: 'white'
            } : {}}
          >
            <FiLayers className={styles.tabIcon} />
            Course Modules
          </button>
          <button
            className={clsx(styles.tab, activeSection === 'hardware' && styles.activeTab)}
            onClick={() => setActiveSection('hardware')}
            style={activeSection === 'hardware' ? { 
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              color: 'white'
            } : {}}
          >
            <FiCpu className={styles.tabIcon} />
            Hardware Guide
          </button>
          <button
            className={clsx(styles.tab, activeSection === 'timeline' && styles.activeTab)}
            onClick={() => setActiveSection('timeline')}
            style={activeSection === 'timeline' ? { 
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              color: 'white'
            } : {}}
          >
            <FiActivity className={styles.tabIcon} />
            Weekly Timeline
          </button>
          <button
            className={clsx(styles.tab, activeSection === 'resources' && styles.activeTab)}
            onClick={() => setActiveSection('resources')}
            style={activeSection === 'resources' ? { 
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              color: 'white'
            } : {}}
          >
            <FiServer className={styles.tabIcon} />
            Resources
          </button>
        </div>

        {/* Content Sections */}
        <div className={styles.contentSections}>
          {activeSection === 'modules' && (
            <div className={styles.contentSection}>
              <div className={styles.sectionHeader}>
                <h2 style={{ color: 'white' }}>Course Modules Breakdown</h2>
                <p className={styles.sectionDescription} style={{ color: '#94A3B8' }}>
                  A comprehensive journey from basic robotics to advanced Physical AI systems
                </p>
              </div>
              <div className={styles.modulesGrid}>
                {modulesData.map((module, index) => (
                  <div key={index} className={styles.moduleCard} style={{ 
                    background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                    border: `2px solid ${module.color}40`,
                    borderTop: `4px solid ${module.color}`
                  }}>
                    <div className={styles.moduleHeader}>
                      <div className={styles.moduleIcon} style={{ background: `${module.color}20`, color: module.color }}>
                        {module.icon}
                      </div>
                      <div>
                        <h3 style={{ color: 'white', marginBottom: '4px' }}>{module.title}</h3>
                        <p style={{ color: module.color, fontSize: '0.9rem' }}>{module.subtitle}</p>
                      </div>
                    </div>
                    <ul className={styles.topicsList}>
                      {module.topics.map((topic, i) => (
                        <li key={i} style={{ color: '#CBD5E1', marginBottom: '8px' }}>
                          <FiCheckCircle style={{ color: module.color, marginRight: '8px' }} />
                          {topic}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>

              {/* Why Physical AI Matters */}
              <div className={styles.infoCard} style={{ 
                background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                borderLeft: '4px solid #8B5CF6'
              }}>
                <h3 style={{ color: 'white', marginBottom: '12px' }}>🎯 Why Physical AI Matters</h3>
                <p style={{ color: '#CBD5E1', lineHeight: '1.7' }}>
                  Humanoid robots excel in our human-centered world because they share our physical form. 
                  They can be trained with abundant data from human interactions, representing a significant 
                  transition from digital AI to <strong style={{ color: '#8B5CF6' }}>embodied intelligence</strong>.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'hardware' && (
            <div className={styles.contentSection}>
              <div className={styles.sectionHeader}>
                <h2 style={{ color: 'white' }}>Hardware Requirements & Architecture</h2>
                <p className={styles.sectionDescription} style={{ color: '#94A3B8' }}>
                  Technical specifications for building your Physical AI lab
                </p>
              </div>
              
              {/* Hardware Tiers */}
              {hardwareTiers.map((tier, index) => (
                <div key={index} className={styles.hardwareTier} style={{ 
                  background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                  borderColor: tier.essential ? '#3B82F6' : '#8B5CF6'
                }}>
                  <div className={styles.tierHeader}>
                    <h3 style={{ color: 'white' }}>{tier.tier}</h3>
                    <span className={styles.tierBadge} style={{ 
                      background: tier.essential ? '#3B82F620' : '#8B5CF620',
                      color: tier.essential ? '#3B82F6' : '#8B5CF6',
                      borderColor: tier.essential ? '#3B82F6' : '#8B5CF6'
                    }}>
                      {tier.essential ? 'Essential' : 'Optional'}
                    </span>
                  </div>
                  <p style={{ color: '#94A3B8', marginBottom: '20px' }}>{tier.description}</p>
                  
                  {tier.components ? (
                    <div className={styles.componentsGrid}>
                      {tier.components.map((comp, idx) => (
                        <div key={idx} className={styles.componentCard} style={{ 
                          background: '#0F172A',
                          borderColor: '#334155'
                        }}>
                          <div className={styles.compIcon} style={{ color: tier.essential ? '#3B82F6' : '#8B5CF6' }}>
                            {comp.icon}
                          </div>
                          <div>
                            <h4 style={{ color: 'white', marginBottom: '4px' }}>{comp.name}</h4>
                            <p style={{ color: '#CBD5E1', fontSize: '0.9rem' }}>{comp.spec}</p>
                            <p style={{ color: '#94A3B8', fontSize: '0.8rem' }}>{comp.note}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.robotOptions}>
                      {tier.options.map((opt, idx) => (
                        <div key={idx} className={styles.robotCard} style={{ 
                          background: '#0F172A',
                          borderColor: idx === 1 ? '#3B82F6' : '#334155'
                        }}>
                          <div className={styles.robotBadge} style={{ 
                            background: idx === 1 ? '#3B82F620' : '#334155',
                            color: idx === 1 ? '#3B82F6' : '#94A3B8'
                          }}>
                            {opt.name}
                          </div>
                          <h4 style={{ color: 'white' }}>{opt.robot}</h4>
                          <div style={{ color: '#F59E0B', fontSize: '1.2rem', fontWeight: 'bold' }}>
                            {opt.price}
                          </div>
                          <p style={{ color: '#94A3B8', fontSize: '0.85rem' }}>{opt.note}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {/* Architecture Summary */}
              <div className={styles.architectureSummary} style={{ 
                background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                border: '2px solid #10B98140'
              }}>
                <h3 style={{ color: 'white', marginBottom: '16px' }}>🏗️ Lab Architecture Summary</h3>
                <div className={styles.archDiagram}>
                  <div className={styles.archNode} style={{ background: '#3B82F620', borderColor: '#3B82F6' }}>
                    <SiUbuntu style={{ color: '#3B82F6' }} />
                    <span>Sim Rig</span>
                    <small>RTX 4080 + Ubuntu</small>
                  </div>
                  <div className={styles.archArrow}>⇄</div>
                  <div className={styles.archNode} style={{ background: '#8B5CF620', borderColor: '#8B5CF6' }}>
                    <RiBrainLine style={{ color: '#8B5CF6' }} />
                    <span>Edge Brain</span>
                    <small>Jetson Orin</small>
                  </div>
                  <div className={styles.archArrow}>→</div>
                  <div className={styles.archNode} style={{ background: '#10B98120', borderColor: '#10B981' }}>
                    <RiRobot2Line style={{ color: '#10B981' }} />
                    <span>Robot</span>
                    <small>Unitree Go2/G1</small>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'timeline' && (
            <div className={styles.contentSection}>
              <div className={styles.sectionHeader}>
                <h2 style={{ color: 'white' }}>13-Week Learning Journey</h2>
                <p className={styles.sectionDescription} style={{ color: '#94A3B8' }}>
                  Structured timeline from foundations to capstone project
                </p>
              </div>

              <div className={styles.timeline}>
                {[
                  { phase: 'Foundation', weeks: '1-2', color: '#3B82F6', topics: ['Physical AI principles', 'Sensor systems', 'Humanoid landscape'] },
                  { phase: 'ROS 2 Core', weeks: '3-5', color: '#8B5CF6', topics: ['ROS 2 architecture', 'Python integration', 'Package development'] },
                  { phase: 'Simulation', weeks: '6-7', color: '#10B981', topics: ['Gazebo physics', 'Unity visualization', 'Sensor simulation'] },
                  { phase: 'NVIDIA Isaac', weeks: '8-10', color: '#76B900', topics: ['Isaac Sim training', 'AI perception', 'Sim-to-real transfer'] },
                  { phase: 'Humanoid Dev', weeks: '11-12', color: '#EF4444', topics: ['Bipedal locomotion', 'Manipulation', 'Human-robot interaction'] },
                  { phase: 'Capstone', weeks: '13', color: '#F59E0B', topics: ['Conversational AI', 'Voice integration', 'Final project'] }
                ].map((item, index) => (
                  <div key={index} className={styles.timelineItem} style={{ borderLeftColor: item.color }}>
                    <div className={styles.timelineMarker} style={{ background: item.color }}></div>
                    <div className={styles.timelineContent}>
                      <div className={styles.timelineHeader}>
                        <h4 style={{ color: 'white' }}>{item.phase}</h4>
                        <span className={styles.weeksBadge} style={{ background: `${item.color}20`, color: item.color }}>
                          Weeks {item.weeks}
                        </span>
                      </div>
                      <ul className={styles.timelineTopics}>
                        {item.topics.map((topic, i) => (
                          <li key={i} style={{ color: '#CBD5E1' }}>{topic}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>

              {/* Assessments */}
              <div className={styles.assessments} style={{ 
                background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                borderColor: '#F59E0B40'
              }}>
                <h3 style={{ color: 'white', marginBottom: '16px' }}>📋 Course Assessments</h3>
                <div className={styles.assessmentGrid}>
                  <div className={styles.assessmentCard}>
                    <div style={{ color: '#3B82F6', fontSize: '2rem' }}>🎯</div>
                    <h4 style={{ color: 'white' }}>ROS 2 Package</h4>
                    <p style={{ color: '#94A3B8' }}>Development project</p>
                  </div>
                  <div className={styles.assessmentCard}>
                    <div style={{ color: '#8B5CF6', fontSize: '2rem' }}>🌐</div>
                    <h4 style={{ color: 'white' }}>Gazebo Simulation</h4>
                    <p style={{ color: '#94A3B8' }}>Environment implementation</p>
                  </div>
                  <div className={styles.assessmentCard}>
                    <div style={{ color: '#10B981', fontSize: '2rem' }}>🧠</div>
                    <h4 style={{ color: 'white' }}>Perception Pipeline</h4>
                    <p style={{ color: '#94A3B8' }}>Isaac-based AI system</p>
                  </div>
                  <div className={styles.assessmentCard}>
                    <div style={{ color: '#F59E0B', fontSize: '2rem' }}>🤖</div>
                    <h4 style={{ color: 'white' }}>Capstone Project</h4>
                    <p style={{ color: '#94A3B8' }}>Autonomous humanoid with conversational AI</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'resources' && (
            <div className={styles.contentSection}>
              <div className={styles.sectionHeader}>
                <h2 style={{ color: 'white' }}>Learning Resources & Options</h2>
                <p className={styles.sectionDescription} style={{ color: '#94A3B8' }}>
                  Choose between on-premise or cloud-native approaches
                </p>
              </div>

              {/* Deployment Options */}
              <div className={styles.deploymentOptions}>
                <div className={styles.optionCard} style={{ 
                  background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                  borderColor: '#3B82F6'
                }}>
                  <div className={styles.optionHeader}>
                    <RiServerLine style={{ color: '#3B82F6', fontSize: '2rem' }} />
                    <h3 style={{ color: 'white' }}>On-Premise Lab</h3>
                    <span className={styles.priceTag} style={{ background: '#3B82F620', color: '#3B82F6' }}>
                      High CapEx
                    </span>
                  </div>
                  <p style={{ color: '#94A3B8' }}>Best for dedicated labs and full control</p>
                  <ul className={styles.optionList}>
                    <li style={{ color: '#CBD5E1' }}>• RTX 4080+ workstations per student</li>
                    <li style={{ color: '#CBD5E1' }}>• Physical Jetson edge kits</li>
                    <li style={{ color: '#CBD5E1' }}>• Direct robot access</li>
                    <li style={{ color: '#CBD5E1' }}>• One-time hardware investment</li>
                  </ul>
                </div>

                <div className={styles.optionCard} style={{ 
                  background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                  borderColor: '#8B5CF6'
                }}>
                  <div className={styles.optionHeader}>
                    <RiCloudLine style={{ color: '#8B5CF6', fontSize: '2rem' }} />
                    <h3 style={{ color: 'white' }}>Cloud-Native Lab</h3>
                    <span className={styles.priceTag} style={{ background: '#8B5CF620', color: '#8B5CF6' }}>
                      High OpEx
                    </span>
                  </div>
                  <p style={{ color: '#94A3B8' }}>Best for rapid deployment and remote access</p>
                  <ul className={styles.optionList}>
                    <li style={{ color: '#CBD5E1' }}>• AWS g5.2xlarge instances ($1.50/hr)</li>
                    <li style={{ color: '#CBD5E1' }}>• NVIDIA Omniverse Cloud</li>
                    <li style={{ color: '#CBD5E1' }}>• Still requires edge kits locally</li>
                    <li style={{ color: '#CBD5E1' }}>• ~$205 per quarter per student</li>
                  </ul>
                </div>
              </div>

              {/* Jetson Economy Kit */}
              <div className={styles.econKit} style={{ 
                background: 'linear-gradient(135deg, #1E293B, #0F172A)',
                borderColor: '#10B981'
              }}>
                <h3 style={{ color: 'white', marginBottom: '16px' }}>💡 Economy Jetson Student Kit</h3>
                <div className={styles.econGrid}>
                  {[
                    { item: 'NVIDIA Jetson Orin Nano', price: '$249', spec: '8GB, 40 TOPS' },
                    { item: 'Intel RealSense D435i', price: '$349', spec: 'RGB-D + IMU' },
                    { item: 'ReSpeaker Mic Array', price: '$69', spec: 'Voice interface' },
                    { item: 'Accessories', price: '$30', spec: 'SD Card, wires' }
                  ].map((item, idx) => (
                    <div key={idx} className={styles.econItem}>
                      <div style={{ color: '#10B981' }}>✓</div>
                      <div>
                        <div style={{ color: 'white' }}>{item.item}</div>
                        <div style={{ color: '#F59E0B', fontWeight: 'bold' }}>{item.price}</div>
                        <div style={{ color: '#94A3B8', fontSize: '0.85rem' }}>{item.spec}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className={styles.totalCost} style={{ borderTopColor: '#334155' }}>
                  <span style={{ color: '#CBD5E1' }}>Total per kit:</span>
                  <span style={{ color: '#F59E0B', fontSize: '1.5rem', fontWeight: 'bold' }}>~$700</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Call to Action */}
        <div className={styles.ctaSection} style={{ 
          background: 'linear-gradient(135deg, #1E293B, #0F172A)',
          border: '2px solid #3B82F640'
        }}>
          <div className={styles.ctaContent}>
            <RiLightbulbFlashLine className={styles.ctaIcon} style={{ color: '#8B5CF6' }} />
            <h2 className={styles.ctaTitle} style={{ color: 'white' }}>Ready to Bridge Digital & Physical AI?</h2>
            <p className={styles.ctaDescription} style={{ color: '#CBD5E1' }}>
              Master the future of AI—where digital intelligence meets physical embodiment. 
              Build systems that understand and interact with the real world.
            </p>
            <div className={styles.ctaButtons}>
              <Link
                className={clsx(styles.primaryButton, styles.button, styles.ctaButton)}
                to="/docs/intro"
                style={{ background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)' }}
              >
                Start Free Course
              </Link>
              <Link
                className={clsx(styles.outlineButton, styles.button, styles.ctaButton)}
                to="/contact"
                style={{ borderColor: '#8B5CF6', color: '#8B5CF6' }}
              >
                Book a Lab Demo
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`${siteConfig.title} - Physical AI & Humanoid Robotics`}
      description="Master Physical AI and Humanoid Robotics with our comprehensive course. Learn to build intelligent machines that bridge digital intelligence with physical embodiment."
    >
      <HomepageHero />
      <main className={styles.mainContainer}>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}