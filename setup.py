#!/usr/bin/env python
# coding: utf-8

import sys
import setuptools
from setuptools import setup


setup_args = dict(
    name="slurmspawner",
    packages=setuptools.find_packages(),
    version='1.0.0',
    description="""Slurmspawner: A spawner for Jupyterhub to spawn notebooks using slurm resource manager.""",
    author="Kamil Burkiewicz",
    url="http://jupyter.org",
    platforms="Linux, Mac OS X",
    python_requires="~=3.5",
    keywords=["Interactive", "Interpreter", "Shell", "Web", "Jupyter"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)

# setuptools requirements
if "setuptools" in sys.modules:
    setup_args["install_requires"] = install_requires = []
    with open("requirements.txt") as f:
        for line in f.readlines():
            req = line.strip()
            if not req or req.startswith(("-e", "#")):
                continue
            install_requires.append(req)


def main():
    setuptools.setup(**setup_args)


if __name__ == "__main__":
    main()

