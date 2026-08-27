from setuptools import find_packages, setup


setup(
    name="adaptive-gcrl",
    version="0.1.0",
    description="Segmentation-invariance stress tests for multi-step offline goal-conditioned reinforcement learning.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
        "rl": ["gymnasium>=0.29", "torch>=2.2"],
        "benchmarks": ["mujoco==3.3.7", "ogbench==1.2.1"],
        "analysis": ["matplotlib>=3.8", "pandas>=2.0", "scipy>=1.11"],
    },
)
