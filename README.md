## Deblock 

Simple deblocking filter by Manao and Fizick. It does a deblocking of the picture, using the deblocking filter of h264.

This version has been back-ported from VapourSynth with high-bit-depth support. All credits go to those who wrote the code.

### Usage
```
Deblock (clip, quant=25, aOffset=0, bOffset=0, planes="yuv")
```
* *quant* - the higher the quant, the stronger the deblocking. It can range from 0 to 60.
* *aOffset* - quant modifier to the blocking detector threshold. Setting it higher means than more edges will deblocked.
* *bOffset* - another quant modifier, for block detecting and for deblocking's strength. There again, the higher, the stronger.
* *planes* - specifies which planes to process between y, u and v.

### MODed by 299792458m
original dll not fast as I expected. so this version was add SIMD opt.  
added parameter
* *opt* - 0:auto  1:c 2:sse4.2 3:AVX2 (only 8-16bit int is SIMDed not float...)

