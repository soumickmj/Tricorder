#!/usr/bin/env python

"""The setup script."""

import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    readme = fh.read()

with open('HISTORY.rst', "r", encoding='utf8') as history_file:
    history = history_file.read()

requirements = [
#    'torch>=1.8'
]

setup_requirements = [
]

tests_requirements = [
]

setuptools.setup(
    name="tricorder", 
    version="0.0.0.3",
    author="Soumick Chatterjee",
    author_email="soumick.chatterjee@ovgu.de",
    description="Collection of small but useful tools for MRI PyTorch",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    url="https://github.com/soumickmj/Tricorder",
    packages=setuptools.find_packages(include=['tricorder', 'tricorder.*']),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=tests_requirements,
    license='MIT license',
    include_package_data=True,
)