#!/bin/bash

if [ command -v git > /dev/null 2>&1 ]; then
  git submodule update --recursive --init
fi

# Library and set management
cd external/hallmark
pip install -e .
cd ../..

# Install pyharm
pip install -e .
