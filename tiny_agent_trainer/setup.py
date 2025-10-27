from setuptools import setup, find_packages
from pathlib import Path

# Get the directory where this setup.py is located
this_directory = Path(__file__).parent

# Try to read README.md, fall back to a default description if it doesn't exist
try:
    long_description = (this_directory.parent / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "A lightweight framework for domain-specific AI agents."

setup(
    name="tiny_agent_trainer",
    version="2.0.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tiny Agent Trainer Team",
    description="A lightweight framework for domain-specific AI agents.",
    # Add project dependencies here
    install_requires=[
        "PyYAML",
        "torch",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'tiny-agent-trainer=tiny_agent_trainer.cli:main',
        ],
    },
)