#!/bin/bash

git submodule update --recursive --init

# Library and set management
cd external/hallmark
pip install -e .
cd ../..

# Install pyharm
pip install -e .
