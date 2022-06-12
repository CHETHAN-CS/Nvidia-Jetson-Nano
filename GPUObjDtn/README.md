# Real-Time-Vehicle-Detection-unsing-Nvidia-jetson-nano

## Follow these steps to create the vehicle detector.


$ sudo apt-get update

$ sudo apt-get install git cmake libpython3-dev python3-numpy

$ git clone --recursive https://github.com/dusty-nv/jetson-inference

$ cd jetson-inference

$ mkdir build

$ cd build

$ cmake ../

## PAUSE A BIT HERE, It will download the model, Select the model you want

## Deselect pythorch (it is not required)

$ make -jS(nproc)

$ sudo make install

$ sudo ldconfig

###############################################################################

################### TO RUN THE PROJECT USE BELOW COMMAND ######################

$ python GPUmain.py

### OR

$ python3 GPUmain.py


###### FROM STUDENTS OF R V COLLEGE OF ENGINEERING


## Source :https://github.com/dusty-nv/jetson-inference
