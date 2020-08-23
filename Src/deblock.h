#pragma once
#define NOMINMAX
#include "avisynth.h"
#include "avs\minmax.h"
#include <Windows.h>
#include <stdint.h>
#include <algorithm>
#include <intrin.h>

#define VS_RESTRICT __restrict
#define VerProc 1
#define HorProc 0

class Deblock : public GenericVideoFilter {
public:
    Deblock(PClip child, int quant, int a_offset, int b_offset, const char* planes, int opt, IScriptEnvironment* env);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
	int __stdcall SetCacheHints(int cachehints, int frame_range);

private:
	template<typename T>
	void deblockHorEdge(T * VS_RESTRICT dstp, const unsigned stride) noexcept;
	template<>
	inline void deblockHorEdge(float * VS_RESTRICT dstp, const unsigned stride) noexcept;
	template<typename T>
	inline void deblockVerEdge(T * VS_RESTRICT dstp, const unsigned stride) noexcept;
	template<>
	inline void deblockVerEdge(float * VS_RESTRICT dstp, const unsigned stride) noexcept;
	template<typename T>
	void Process(PVideoFrame &dst, IScriptEnvironment *env) noexcept;
	template<typename T>
	void Process(PVideoFrame &dst, int plane, IScriptEnvironment *env) noexcept;
	
    template<typename T>
    inline void deblockEdgeOPT(T* VS_RESTRICT dstp, const unsigned stride,int mode);
    template<typename T>
    inline void deblockEdgeOPT(uint8_t * VS_RESTRICT dstp, const unsigned stride, int mode);
    template<typename T>
    inline void deblockEdgeOPT(uint16_t * VS_RESTRICT dstp, const unsigned stride, int mode);

    template<typename T>
    inline void deblockEdgeOPT8(T* VS_RESTRICT dstp, const unsigned stride, int mode);
    template<>
    inline void deblockEdgeOPT8(uint8_t* VS_RESTRICT dstp, const unsigned stride, int mode);
    template<>
    inline void deblockEdgeOPT8(uint16_t* VS_RESTRICT dstp, const unsigned stride, int mode);

    template<typename T>
    inline void deblockVerEdgeOPT(T* VS_RESTRICT dstp, const unsigned stride) ;

    inline void deblockEdgeOPT_cal8_sse4(__m128i& sp2_16, __m128i& sp1_16, __m128i& sp0_16, __m128i& sq0_16, __m128i& sq1_16, __m128i& sq2_16);

	bool _process[3];
	int _alpha, _beta, _c0, _c1;
	float _alphaF, _betaF, _c0F, _c1F;
	int _peak;
	int _opt;
};

static inline void transpose_8bit_4x4(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
    int& out0, int& out1, int& out2, int& out3) {
    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    const __m128i a0 = _mm_unpacklo_epi8(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi8(in2, in3);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    out0 = _mm_extract_epi32(b0, 0);
    out1 = _mm_extract_epi32(b0, 1);
    out2 = _mm_extract_epi32(b0, 2);
    out3 = _mm_extract_epi32(b0, 3);
}
static inline void transpose_8bit_6x4(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
    __m128i& out0, __m128i& out1, __m128i& out2, __m128i& out3, __m128i& out4, __m128i& out5) {
    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    const __m128i a0 = _mm_unpacklo_epi8(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi8(in2, in3);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    // b1: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpackhi_epi16(a0, a1);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    // out[4]: 04 14 24 34
    // out[5]: 05 15 25 35
    //下位4byteしか正しくないので注意
    out0 = b0;
    out1 = _mm_srli_si128(b0, 4);
    out2 = _mm_srli_si128(b0, 8);
    out3 = _mm_srli_si128(b0, 12);
    out4 = b1;
    out5 = _mm_srli_si128(b1, 4);
}

static inline void transpose_8bit_8x4(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
    int& out0, int& out1, int& out2, int& out3, int& out4, int& out5, int& out6, int& out7) {
    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    const __m128i a0 = _mm_unpacklo_epi8(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi8(in2, in3);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    // b1: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpackhi_epi16(a0, a1);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    // out[4]: 04 14 24 34
    // out[5]: 05 15 25 35
    // out[6]: 06 16 26 36
    // out[7]: 07 17 27 37
    out0 = _mm_cvtsi128_si32(b0);
    out1 = _mm_cvtsi128_si32(_mm_srli_si128(b0, 4));
    out2 = _mm_cvtsi128_si32(_mm_srli_si128(b0, 8));
    out3 = _mm_cvtsi128_si32(_mm_srli_si128(b0, 12));
    out4 = _mm_cvtsi128_si32(b1);
    out5 = _mm_cvtsi128_si32(_mm_srli_si128(b1, 4));
    out6 = _mm_cvtsi128_si32(_mm_srli_si128(b1, 8));
    out7 = _mm_cvtsi128_si32(_mm_srli_si128(b1, 12));
}
//作ったけど使えなかった・・・
static inline void transpose_8bit_4x4x2(__m128i in0, __m128i in1, __m128i in2, __m128i in3,
    __m128i& out0, __m128i& out1, __m128i& out2, __m128i& out3) {

    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    const __m128i a0 = _mm_unpacklo_epi8(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi8(in2, in3);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    // b1: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpackhi_epi16(a0, a1);

    // Unpack 32 bit elements resulting in:
    // c0:00 10 20 30 04 14 24 34 02 12 22 32 06 16 26 36 
    // c1:01 11 21 31 05 15 25 35 03 13 23 33 07 17 27 37
    const __m128i c0 = _mm_unpacklo_epi32(b0, b1);
    const __m128i c1 = _mm_unpackhi_epi32(b0, b1);

    // Unpack 64 bit elements resulting in:
    // out[0]:00 10 20 30 04 14 24 34
    // out[1]:01 11 21 31 05 15 25 35
    // out[2]:02 12 22 32 06 16 26 36
    // out[3]:03 13 23 33 07 17 27 37
    out0 = c0;
    out1 = c1;
    out2 = _mm_srli_si128(c0, 8);
    out3 = _mm_srli_si128(c1, 8);
}
static inline void transpose_8bit_6x8(__m128i in0, __m128i in1, __m128i in2, __m128i in3, __m128i in4, __m128i in5, __m128i in6, __m128i in7,
    __m128i& out0, __m128i& out1, __m128i& out2, __m128i& out3, __m128i& out4, __m128i& out5) {

    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // in[4]: 40 41 42 43 44 45 46 47
    // in[5]: 50 51 52 53 54 55 56 57
    // in[6]: 60 61 62 63 64 65 66 67
    // in[7]: 70 71 72 73 74 75 76 77
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    // a2:    40 50 41 51 42 52 43 53  44 54 45 55 46 56 47 57
    // a3:    60 70 61 71 62 72 63 73  64 74 65 75 66 76 67 77
    const __m128i a0 = _mm_unpacklo_epi8(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi8(in2, in3);
    const __m128i a2 = _mm_unpacklo_epi8(in4, in5);
    const __m128i a3 = _mm_unpacklo_epi8(in6, in7);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    // b1: 40 50 60 70 41 51 61 71  42 52 62 72 43 53 63 73
    // b2: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
    // b3: 44 54 64 74 45 55 65 75  46 56 66 76 47 57 67 77
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpackhi_epi16(a0, a1);
    const __m128i b2 = _mm_unpacklo_epi16(a2, a3);
    const __m128i b3 = _mm_unpackhi_epi16(a2, a3);

    // Unpack 32 bit elements resulting in:
    // c0: 00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71
    // c1: 02 12 22 32 42 52 62 72  03 13 23 33 43 53 63 73
    // c2: 04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75
    // c3: 06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77
    const __m128i c0 = _mm_unpacklo_epi32(b0, b2);
    const __m128i c1 = _mm_unpackhi_epi32(b0, b2);
    const __m128i c2 = _mm_unpacklo_epi32(b1, b3);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30 40 50 60 70
    // out[1]: 01 11 21 31 41 51 61 71
    // out[2]: 02 12 22 32 42 52 62 72
    // out[3]: 03 13 23 33 43 53 63 73
    // out[4]: 04 14 24 34 44 54 64 74
    // out[5]: 05 15 25 35 45 55 65 75
    out0 = c0;
    out1 = _mm_srli_si128(c0, 8);
    out2 = c1;
    out3 = _mm_srli_si128(c1, 8);
    out4 = c2;
    out5 = _mm_srli_si128(c2, 8);
}
static inline void transpose_16bit_6x4(
    __m128i in0 , __m128i in1, __m128i in2, __m128i in3,
    __m128i &out0, __m128i &out1,__m128i &out2,__m128i &out3,__m128i &out4,__m128i &out5) {
    // Unpack 16 bit elements. Goes from:
    // in[0]: 00 01 02 03  04 05 06 07
    // in[1]: 10 11 12 13  14 15 16 17
    // in[2]: 20 21 22 23  24 25 26 27
    // in[3]: 30 31 32 33  34 35 36 37

    // to:
    // a0:    00 10 01 11  02 12 03 13
    // a1:    20 30 21 31  22 32 23 33
    // a4:    04 14 05 15  06 16 07 17
    // a5:    24 34 25 35  26 36 27 37
    const __m128i a0 = _mm_unpacklo_epi16(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi16(in2, in3);
    const __m128i a4 = _mm_unpackhi_epi16(in0, in1);
    const __m128i a5 = _mm_unpackhi_epi16(in2, in3);

    // Unpack 32 bit elements resulting in:
    // b0: 00 10 20 30  01 11 21 31
    // b2: 04 14 24 34  05 15 25 35
    // b4: 02 12 22 32  03 13 23 33
    // b6: 06 16 26 36  07 17 27 37
    const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
    const __m128i b2 = _mm_unpacklo_epi32(a4, a5);
    const __m128i b4 = _mm_unpackhi_epi32(a0, a1);
    //const __m128i b6 = _mm_unpackhi_epi32(a4, a5);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30  XX XX XX XX
    // out[1]: 01 11 21 31  XX XX XX XX
    // out[2]: 02 12 22 32  XX XX XX XX
    // out[3]: 03 13 23 33  XX XX XX XX
    // out[4]: 04 14 24 34  XX XX XX XX
    // out[5]: 05 15 25 35  XX XX XX XX
    const __m128i zeros = _mm_setzero_si128();
    out0 = _mm_unpacklo_epi64(b0, zeros);
    out1 = _mm_unpackhi_epi64(b0, zeros);
    out2 = _mm_unpacklo_epi64(b4, zeros);
    out3 = _mm_unpackhi_epi64(b4, zeros);
    out4 = _mm_unpacklo_epi64(b2, zeros);
    out5 = _mm_unpackhi_epi64(b2, zeros);
}

static inline void transpose_16bit_4x4(
    __m128i in0, __m128i in1, __m128i in2, __m128i in3,
    __m128i &out0, __m128i &out1, __m128i &out2, __m128i &out3) {
    // Unpack 16 bit elements. Goes from:
    // in[0]: 00 01 02 03  XX XX XX XX
    // in[1]: 10 11 12 13  XX XX XX XX
    // in[2]: 20 21 22 23  XX XX XX XX
    // in[3]: 30 31 32 33  XX XX XX XX
    // to:
    // a0:    00 10 01 11  02 12 03 13
    // a1:    20 30 21 31  22 32 23 33
    const __m128i a0 = _mm_unpacklo_epi16(in0, in1);
    const __m128i a1 = _mm_unpacklo_epi16(in2, in3);

    // Unpack 32 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    out0 = _mm_unpacklo_epi32(a0, a1);
    out1 = _mm_srli_si128(out0, 8);
    out2 = _mm_unpackhi_epi32(a0, a1);
    out3 = _mm_srli_si128(out2, 8);
}
