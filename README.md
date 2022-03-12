# tensorflow_tutorials

## Environments
```shell
sysctl -a | grep machdep.cpu.brand_string
machdep.cpu.brand_string: Apple M1 Max

sw_vers
ProductName:	macOS
ProductVersion:	12.2.1
BuildVersion:	21D62

git --version
git version 2.29.2

pyenv --version
pyenv 2.2.3

python --version
Python 3.8.2
```

## M1 mac
```shell
# Setting up
git clone https://github.com/tomnt/tensorflow_tutorials.git
cd tensorflow_tutorials
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

source Miniforge3-MacOSX-arm64.sh
# >>> miniforge3
source miniforge3/bin/activate 
conda install -c apple tensorflow-deps==2.6.0

# requirements.txt
python -m pip install matplotlib==3.5.1
python -m pip install tensorflow-macos==2.8.0

# Running the script
python keras_classification.py
```

### References
Getting Started with tensorflow-metal PluggableDevice
https://developer.apple.com/metal/tensorflow-plugin/

## All other CPUs
```
# Setting up
git clone https://github.com/tomnt/tensorflow_tutorials.git
cd tensorflow_tutorials
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt

# Running the script
python3 keras_classification.py
```
