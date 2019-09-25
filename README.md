# DeepEyePredictor
DeepEye Predictor.

* CUDA 10.1
* cuDNN 7.6.3

## Windows 10
* Visual Studio 2017

### download library
#### CodeMeter Runtime
http://www.suncarla.co.jp/codemeter/runtimekit/v680/index.html

* CodeMeter Runtime Kit Ver 6.80 (x64)

#### OpenCV
opencv 3.4.3
https://opencv.org/releases/

```
C:\opencv
```

## Ubuntu 18.04

### download library
#### CodeMeter Runtime
http://www.suncarla.co.jp/codemeter/runtimekit/v680/index.html

* CodeMeter Runtime Kit Ver 6.80 (x64)


### build
* CMake

```bash
$ cd src/classifier_sample or ssd_sample
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```


# Download weights file
https://g5-compmind.s-portcloud.com/fc2bbf4aa4554596df0bcd35b515b1db453cf94f8

pass: VC1ZvmB6

```
models/googlenet_lego.deep
models/ssd_10classes.deep
```