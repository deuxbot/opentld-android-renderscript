# Title

> opentld-android-renderscript

This is a port of the OpenTLD algorithm to Android using RenderScript to improve the speed.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Background

The OpenTLD was originally published in [MATLAB](https://github.com/zk00006/OpenTLD)  by Zdenek Kalaland later in [C++](https://github.com/gnebehay/OpenTLD) by Georg Nebehay. This code is based on the C++ implementation by Georg Nebehay and is inspired in the [Android](https://github.com/trandi/OpenTLDAndroid) port by Dan Oprescu.

## Install

No instructions are provided. This code was developed back in 2014 with Eclipse ADT Plugin and OpenCV 2.4.8. Further modifications may be required to work in current systems and this project is no longer maintained.

## Usage
Create a bounding box arround and object and the algorithm will start. See the demo:

[![](https://i.ytimg.com/vi/CSKyr6it4Qw/1.jpg)](https://youtu.be/CSKyr6it4Qw) 

## Results

The overall performance was improved by a factor of  2.35 while the optimised phases were improved by a factor of 8.26, reaching a factor 22.96 in one phase.  Tests were measured in a Samsung Galaxy S3 with a window resolution of 320x240 (QVGA). 


## License

See [LICENSE](https://github.com/deuxbot/opentld-android-renderscript/license)