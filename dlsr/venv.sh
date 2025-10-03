#!/bin/bash

set -e

# apt-get install python3-tk -y

python3 -m venv venv

source venv/bin/activate

python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt

echo -e "\033[1;32m[Environment set up successfully. Use \"source venv/bin/activate\" to enter]\033[0m"