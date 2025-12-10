from setuptools import setup, find_packages

setup(
    name="physical-ai-book",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "gymnasium==0.26.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "omegaconf>=2.2.0",
        "pybullet>=3.2.5",
        "isaac-sim",
        "better-auth",
        "qdrant-client",
        "google-generativeai"
    ],
    entry_points={
        'console_scripts': [
            'physical-ai-book=main:main',
        ],
    },
    author="Robotics Research Team",
    author_email="robotics@example.com",
    description="Interactive textbook for Physical AI and Humanoid Robotics",
    python_requires=">=3.8",
)
