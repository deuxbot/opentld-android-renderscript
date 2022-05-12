# opentld-android-renderscript

> Fast OpenTLD port to Android with RenderScript


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Background

The OpenTLD was originally published in [MATLAB](https://github.com/zk00006/OpenTLD)  by Zdenek Kalaland and later in [C++](https://github.com/gnebehay/OpenTLD) by Georg Nebehay. This code is based in the C++ implementation by Georg Nebehay and is inspired in the [Android](https://github.com/trandi/OpenTLDAndroid) port by Dan Oprescu.

## Install

No instructions are provided. This code was developed back in 2014 with Eclipse ADT Plugin and OpenCV 2.4.8. Further modifications may be required to work in current systems and this project is no longer maintained.

## Usage
Create a bounding box arround and object and the algorithm will start. See the demo:

[![](https://i.ytimg.com/vi/4itJTOFhzss/hqdefault.jpg?s%E2%80%A6AFwAcABBg==&rs=AOn4CLDjJMeUj7maZzZ438BEtSAHRIHMCg)](https://youtu.be/4itJTOFhzss) 

## Results

Overall performance was improved by a factor 2.35 while the optimised phases were improved by a factor 8.26, reaching a factor 22.96 in one phase. Tests were measured on a Samsung Galaxy S3 with a frame resolution of 320x240 (QVGA). 


## License

See [LICENSE](https://github.com/deuxbot/opentld-android-renderscript/blob/master/LICENSE)
