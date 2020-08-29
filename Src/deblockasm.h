#pragma once

	//template<>
	void deblockEdgeOPT(uint8_t* VS_RESTRICT dstp, const unsigned stride, int mode);
	void deblockEdgeOPT(uint16_t* VS_RESTRICT dstp, const unsigned stride, int mode);

	//template<typename T>
	void deblockEdgeOPT8(uint8_t* VS_RESTRICT dstp, const unsigned stride, int mode);
	void deblockEdgeOPT8(uint16_t* VS_RESTRICT dstp, const unsigned stride, int mode);

	void deblockEdgeOPT_cal8_sse4(__m128i& sp2_16, __m128i& sp1_16, __m128i& sp0_16, __m128i& sq0_16, __m128i& sq1_16, __m128i& sq2_16);
	void deblockEdgeOPT_cal16_AVX2(__m128i& sp2_16, __m128i& sp1_16, __m128i& sp0_16, __m128i& sq0_16, __m128i& sq1_16, __m128i& sq2_16);

