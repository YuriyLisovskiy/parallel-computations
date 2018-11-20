### OpenCL Task

#### Install libraries
Generic ubuntu packages for OpenCL:
```bash
$ sudo apt-get install ocl-icd-libopencl1
$ sudo apt-get install opencl-headers
$ sudo apt-get install clinfo
```

Package that allows to compile OpenCL code:
```bash
$ sudo apt-get install ocl-icd-opencl-dev
```

Package that enables runnig openCL on Intel GT, IvyBridge and up:
```bash
$ sudo apt install beignet
```

For more info about the installation, read [this](https://askubuntu.com/questions/850281/opencl-on-ubuntu-16-04-intel-sandy-bridge-cpu/850594) article.

#### Build and Run (Ubuntu 16.04 LTS)
```bash
$ make
```
