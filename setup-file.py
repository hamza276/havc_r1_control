from setuptools import setup, find_packages

setup(
    name="hvac_rl_control",
    version="0.1.0",
    packages=find_packages(),
    
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.28.1",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "tensorboard>=2.6.0"
    ],
    
    author="Your Name",
    author_email="your.email@example.com",
    description="Reinforcement Learning-based HVAC Control System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.8",
)