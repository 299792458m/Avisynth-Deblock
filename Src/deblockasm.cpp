#include "deblock.h"
#include "deblockasm.h"

inline void deblockEdgeOPT_cal8_sse4(__m128i& sp2_16, __m128i& sp1_16, __m128i& sp0_16, __m128i& sq0_16, __m128i& sq1_16, __m128i& sq2_16)
{
	const int alpha = Deblock._alpha;
	const int beta = Deblock::_beta;
	const int c0 = Deblock::_c0;
	const int c1 = Deblock::_c1;

	//for (unsigned i = 0; i < 4; i++) {
		//if (std::abs(sp0[i] - sq0[i]) < alpha && std::abs(sp1[i] - sp0[i]) < beta && std::abs(sq0[i] - sq1[i]) < beta) {

	sp1_16 = _mm_cvtepu8_epi16(sp1_16);
	sp0_16 = _mm_cvtepu8_epi16(sp0_16);	//SSE4.1
	sq0_16 = _mm_cvtepu8_epi16(sq0_16);
	sq1_16 = _mm_cvtepu8_epi16(sq1_16);

	auto tsub = _mm_sub_epi16(sp0_16, sq0_16);
	auto tabs = _mm_abs_epi16(tsub);			//SSSE3 minmaxでやろうと思ったがmin/max_epi32はSSE4.1...どちらにしろ4.1だが、なら普通にabs取ればよい
	auto mask1 = _mm_cmplt_epi16(tabs, _mm_set1_epi16(alpha));	//asmではオペランドを入れ替えてgtになるらしい

	tsub = _mm_sub_epi16(sp1_16, sp0_16);
	tabs = _mm_abs_epi16(tsub);
	auto mask2 = _mm_cmplt_epi16(tabs, _mm_set1_epi16(beta));

	tsub = _mm_sub_epi16(sq0_16, sq1_16);
	tabs = _mm_abs_epi16(tsub);
	auto mask3 = _mm_cmplt_epi16(tabs, _mm_set1_epi16(beta));

	auto mask = _mm_and_si128(mask1, mask2);
	mask = _mm_and_si128(mask, mask3);	//ifはFF

	//if (~_mm_test_all_zeros(mask, mask))	//あってもなくてもあまり速度は変わらない
	{

		//	const int ap = std::abs(sp2[i] - sp0[i]);
		//	const int aq = std::abs(sq2[i] - sq0[i]);

		sp2_16 = _mm_cvtepu8_epi16(sp2_16);
		sq2_16 = _mm_cvtepu8_epi16(sq2_16);

		tsub = _mm_sub_epi16(sp2_16, sp0_16);
		auto ap_16 = _mm_abs_epi16(tsub);

		tsub = _mm_sub_epi16(sq2_16, sq0_16);
		auto aq_16 = _mm_abs_epi16(tsub);


		//int c = c0;
		//if (ap < beta)
		//	c += c1;
		//if (aq < beta)
		//	c += c1;

		auto c_16 = _mm_set1_epi16(c0);
		auto c1_16 = _mm_set1_epi16(c1);
		auto mask4 = _mm_cmplt_epi16(ap_16, _mm_set1_epi16(beta));
		auto temp = _mm_and_si128(c1_16, mask4);
		c_16 = _mm_add_epi16(c_16, temp);

		auto mask5 = _mm_cmplt_epi16(aq_16, _mm_set1_epi16(beta));
		temp = _mm_and_si128(c1_16, mask5);	//mask4と5の順番注意・・・元のcではqが先になっているが、逆にしておく↑のcも逆にした
		c_16 = _mm_add_epi16(c_16, temp);
		auto c_16m = _mm_sub_epi16(_mm_setzero_si128(), c_16);

		//const int avg = (sp0[i] + sq0[i] + 1) >> 1;
		//const int delta = min(max(((sq0[i] - sp0[i]) * 4 + sp1[i] - sq1[i] + 4) >> 3, -c), c);
		//const int deltap1 = min(max((sp2[i] + avg - sp1[i] * 2) >> 1, -c0), c0);
		//const int deltaq1 = min(max((sq2[i] + avg - sq1[i] * 2) >> 1, -c0), c0);

		auto avg_16 = _mm_add_epi16(sp0_16, sq0_16);
		avg_16 = _mm_add_epi16(avg_16, _mm_set1_epi16(1));
		avg_16 = _mm_srai_epi16(avg_16, 1);

		auto delta_16 = _mm_sub_epi16(sq0_16, sp0_16);
		delta_16 = _mm_slli_epi16(delta_16, 2);
		delta_16 = _mm_add_epi16(delta_16, sp1_16);
		delta_16 = _mm_sub_epi16(delta_16, sq1_16);
		delta_16 = _mm_add_epi16(delta_16, _mm_set1_epi16(4));
		delta_16 = _mm_srai_epi16(delta_16, 3);
		delta_16 = _mm_max_epi16(delta_16, c_16m);
		delta_16 = _mm_min_epi16(delta_16, c_16);

		auto deltap1_16 = _mm_slli_epi16(sp1_16, 1);
		deltap1_16 = _mm_sub_epi16(sp2_16, deltap1_16);
		deltap1_16 = _mm_add_epi16(deltap1_16, avg_16);
		deltap1_16 = _mm_srai_epi16(deltap1_16, 1);
		deltap1_16 = _mm_max_epi16(deltap1_16, _mm_set1_epi16(-c0));
		deltap1_16 = _mm_min_epi16(deltap1_16, _mm_set1_epi16(c0));

		auto deltaq1_16 = _mm_slli_epi16(sq1_16, 1);
		deltaq1_16 = _mm_sub_epi16(sq2_16, deltaq1_16);
		deltaq1_16 = _mm_add_epi16(deltaq1_16, avg_16);
		deltaq1_16 = _mm_srai_epi16(deltaq1_16, 1);
		deltaq1_16 = _mm_max_epi16(deltaq1_16, _mm_set1_epi16(-c0));
		deltaq1_16 = _mm_min_epi16(deltaq1_16, _mm_set1_epi16(c0));

		//sp0[i] = min(max(sp0[i] + delta, 0), _peak);
		//sq0[i] = min(max(sq0[i] - delta, 0), _peak);

		//全部ストアする
		temp = _mm_and_si128(mask, delta_16);
		sp0_16 = _mm_add_epi16(sp0_16, temp);
		sp0_16 = _mm_max_epi16(sp0_16, _mm_setzero_si128());
		sp0_16 = _mm_min_epi16(sp0_16, _mm_set1_epi16(_peak));
		sp0_16 = _mm_packus_epi16(sp0_16, _mm_setzero_si128());

		sq0_16 = _mm_sub_epi16(sq0_16, temp);
		sq0_16 = _mm_max_epi16(sq0_16, _mm_setzero_si128());
		sq0_16 = _mm_min_epi16(sq0_16, _mm_set1_epi16(_peak));
		sq0_16 = _mm_packus_epi16(sq0_16, _mm_setzero_si128());

		//if (ap < beta)
		//	sp1[i] += deltap1;
		//if (aq < beta)
		//	sq1[i] += deltaq1;

		temp = _mm_and_si128(mask4, deltap1_16);	//ifの
		temp = _mm_and_si128(mask, temp);			//分岐に入るかどうか
		sp1_16 = _mm_add_epi16(sp1_16, temp);
		sp1_16 = _mm_packus_epi16(sp1_16, _mm_setzero_si128());

		temp = _mm_and_si128(mask5, deltaq1_16);	//ifの
		temp = _mm_and_si128(mask, temp);			//分岐に入るかどうか
		sq1_16 = _mm_add_epi16(sq1_16, temp);
		sq1_16 = _mm_packus_epi16(sq1_16, _mm_setzero_si128());
	}
}


inline void deblockEdgeOPT(uint8_t* VS_RESTRICT dstp, const unsigned stride, int mode) {

	const uint8_t* sp2;
	uint8_t* VS_RESTRICT sp1;
	uint8_t* VS_RESTRICT sp0;
	uint8_t* VS_RESTRICT sq0;
	uint8_t* VS_RESTRICT sq1;
	const uint8_t* sq2;

	__m128i sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16;

	if (mode == HorProc) {	//水平モード
		sp2 = dstp - stride * 3;
		sp1 = dstp - stride * 2;
		sp0 = dstp - stride;
		sq0 = dstp;
		sq1 = dstp + stride;
		sq2 = dstp + stride * 2;

		sp2_16 = _mm_cvtsi32_si128(*(int*)sp2);	//本来はifの後でよい
		sp1_16 = _mm_cvtsi32_si128(*(int*)sp1);	//4個分
		sp0_16 = _mm_cvtsi32_si128(*(int*)sp0);
		sq0_16 = _mm_cvtsi32_si128(*(int*)sq0);
		sq1_16 = _mm_cvtsi32_si128(*(int*)sq1);
		sq2_16 = _mm_cvtsi32_si128(*(int*)sq2);	//本来はifの後でよい

	}
	else {		//垂直モードの時は転置処理
		sp1 = dstp - 3;
		sp0 = dstp - 3 + stride;
		sq0 = dstp - 3 + stride * 2;
		sq1 = dstp - 3 + stride * 3;

		auto data0 = _mm_loadu_si64((__m128i*)sp1);	//読み込みは8byte分(使うのは6byte)
		auto data1 = _mm_loadu_si64((__m128i*)sp0);
		auto data2 = _mm_loadu_si64((__m128i*)sq0);
		auto data3 = _mm_loadu_si64((__m128i*)sq1);

		transpose_8bit_6x4(data0, data1, data2, data3,
			sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);
	}

	Deblock::deblockEdgeOPT_cal8_sse4(sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);

	if (mode == HorProc) {	//水平モード
		*(uint32_t*)sp1 = _mm_cvtsi128_si32(sp1_16);
		*(uint32_t*)sp0 = _mm_cvtsi128_si32(sp0_16);
		*(uint32_t*)sq0 = _mm_cvtsi128_si32(sq0_16);
		*(uint32_t*)sq1 = _mm_cvtsi128_si32(sq1_16);
	}
	else {		//垂直モードの時は転置処理
		int out0, out1, out2, out3;
		//transpose_8bit_4x4x2(sp1_16, sp0_16, sq0_16, sq1_16,out0, out1, out2, out3);	//4x4x2でやろうと思ったけど1つ目の結果(q1)がないと2つ目の計算ができないことに気が付いた・・・orz
		//処理中のsp0はdstp[-3]で、出力先はdstp[-2]から
		transpose_8bit_4x4(sp1_16, sp0_16, sq0_16, sq1_16, out0, out1, out2, out3);
		*(uint32_t*)(sp1 + 1) = out0;
		*(uint32_t*)(sp0 + 1) = out1;
		*(uint32_t*)(sq0 + 1) = out2;
		*(uint32_t*)(sq1 + 1) = out3;
	}
}



inline void deblockEdgeOPT(uint16_t* VS_RESTRICT dstp, const unsigned stride, int mode) {
	const int alpha = _alpha;
	const int beta = _beta;
	const int c0 = _c0;
	const int c1 = _c1;

	const uint16_t* sp2;
	uint16_t* VS_RESTRICT sp1;
	uint16_t* VS_RESTRICT sp0;
	uint16_t* VS_RESTRICT sq0;
	uint16_t* VS_RESTRICT sq1;
	const uint16_t* sq2;

	__m128i sp2_32, sp1_32, sp0_32, sq0_32, sq1_32, sq2_32;

	if (mode == HorProc) {	//水平モード
		sp2 = dstp - stride * 3;
		sp1 = dstp - stride * 2;
		sp0 = dstp - stride;
		sq0 = dstp;
		sq1 = dstp + stride;
		sq2 = dstp + stride * 2;

		sp2_32 = _mm_loadu_si64((__m128i*)sp2);	//本来はifの後でよい
		sp1_32 = _mm_loadu_si64((__m128i*)sp1);
		sp0_32 = _mm_loadu_si64((__m128i*)sp0);
		sq0_32 = _mm_loadu_si64((__m128i*)sq0);
		sq1_32 = _mm_loadu_si64((__m128i*)sq1);
		sq2_32 = _mm_loadu_si64((__m128i*)sq2);	//本来はifの後でよい
	}
	else {		//垂直モードの時は転置処理
		//sp2 = dstp -2 - stride * 3;
		sp1 = dstp - 3;
		sp0 = dstp - 3 + stride;
		sq0 = dstp - 3 + stride * 2;
		sq1 = dstp - 3 + stride * 3;
		//sq2 = dstp + stride * 2;

		auto data0 = _mm_loadu_si128((__m128i*)sp1);	//読み込みは8ではなく16byte分(使うのは12byte)
		auto data1 = _mm_loadu_si128((__m128i*)sp0);
		auto data2 = _mm_loadu_si128((__m128i*)sq0);
		auto data3 = _mm_loadu_si128((__m128i*)sq1);

		transpose_16bit_6x4(data0, data1, data2, data3,
			sp2_32, sp1_32, sp0_32, sq0_32, sq1_32, sq2_32);
	}



	//for (unsigned i = 0; i < 4; i++) {
		//if (std::abs(sp0[i] - sq0[i]) < alpha && std::abs(sp1[i] - sp0[i]) < beta && std::abs(sq0[i] - sq1[i]) < beta) {

	sp0_32 = _mm_cvtepu16_epi32(sp0_32);	//SSE4.1
	sp1_32 = _mm_cvtepu16_epi32(sp1_32);
	sq0_32 = _mm_cvtepu16_epi32(sq0_32);
	sq1_32 = _mm_cvtepu16_epi32(sq1_32);

	auto tsub = _mm_sub_epi32(sp0_32, sq0_32);
	auto tabs = _mm_abs_epi32(tsub);			//SSSE3 minmaxでやろうと思ったがmin/max_epi32はSSE4.1...どちらにしろ4.1だが、なら普通にabs取ればよい
	auto mask1 = _mm_cmplt_epi32(tabs, _mm_set1_epi32(alpha));	//asmではオペランドを入れ替えてgtになるらしい

	tsub = _mm_sub_epi32(sp1_32, sp0_32);
	tabs = _mm_abs_epi32(tsub);
	auto mask2 = _mm_cmplt_epi32(tabs, _mm_set1_epi32(beta));

	tsub = _mm_sub_epi32(sq0_32, sq1_32);
	tabs = _mm_abs_epi32(tsub);
	auto mask3 = _mm_cmplt_epi32(tabs, _mm_set1_epi32(beta));

	auto mask = _mm_and_si128(mask1, mask2);
	mask = _mm_and_si128(mask, mask3);	//ifはFF

	//if (~_mm_test_all_zeros(mask, mask))	//あってもなくてもあまり速度は変わらない？
	{

		//	const int ap = std::abs(sp2[i] - sp0[i]);
		//	const int aq = std::abs(sq2[i] - sq0[i]);

		sp2_32 = _mm_cvtepu16_epi32(sp2_32);
		sq2_32 = _mm_cvtepu16_epi32(sq2_32);

		tsub = _mm_sub_epi32(sp2_32, sp0_32);
		auto ap_32 = _mm_abs_epi32(tsub);

		tsub = _mm_sub_epi32(sq2_32, sq0_32);
		auto aq_32 = _mm_abs_epi32(tsub);


		//int c = c0;
		//if (ap < beta)
		//	c += c1;
		//if (aq < beta)
		//	c += c1;

		auto c_32 = _mm_set1_epi32(c0);
		auto c1_32 = _mm_set1_epi32(c1);
		auto mask4 = _mm_cmplt_epi32(ap_32, _mm_set1_epi32(beta));
		auto temp = _mm_and_si128(c1_32, mask4);
		c_32 = _mm_add_epi32(c_32, temp);

		auto mask5 = _mm_cmplt_epi32(aq_32, _mm_set1_epi32(beta));
		temp = _mm_and_si128(c1_32, mask5);	//mask4と5の順番注意・・・元のcではqが先になっているが、逆にしておく↑のcも逆にした
		c_32 = _mm_add_epi32(c_32, temp);
		auto c_32m = _mm_sub_epi32(_mm_setzero_si128(), c_32);

		//const int avg = (sp0[i] + sq0[i] + 1) >> 1;
		//const int delta = min(max(((sq0[i] - sp0[i]) * 4 + sp1[i] - sq1[i] + 4) >> 3, -c), c);
		//const int deltap1 = min(max((sp2[i] + avg - sp1[i] * 2) >> 1, -c0), c0);
		//const int deltaq1 = min(max((sq2[i] + avg - sq1[i] * 2) >> 1, -c0), c0);

		auto avg_32 = _mm_add_epi32(sp0_32, sq0_32);
		avg_32 = _mm_add_epi32(avg_32, _mm_set1_epi32(1));
		avg_32 = _mm_srai_epi32(avg_32, 1);

		auto delta_32 = _mm_sub_epi32(sq0_32, sp0_32);
		delta_32 = _mm_slli_epi32(delta_32, 2);
		delta_32 = _mm_add_epi32(delta_32, sp1_32);
		delta_32 = _mm_sub_epi32(delta_32, sq1_32);
		delta_32 = _mm_add_epi32(delta_32, _mm_set1_epi32(4));
		delta_32 = _mm_srai_epi32(delta_32, 3);
		delta_32 = _mm_max_epi32(delta_32, c_32m);
		delta_32 = _mm_min_epi32(delta_32, c_32);

		auto deltap1_32 = _mm_slli_epi32(sp1_32, 1);
		deltap1_32 = _mm_sub_epi32(sp2_32, deltap1_32);
		deltap1_32 = _mm_add_epi32(deltap1_32, avg_32);
		deltap1_32 = _mm_srai_epi32(deltap1_32, 1);
		deltap1_32 = _mm_max_epi32(deltap1_32, _mm_set1_epi32(-c0));
		deltap1_32 = _mm_min_epi32(deltap1_32, _mm_set1_epi32(c0));

		auto deltaq1_32 = _mm_slli_epi32(sq1_32, 1);
		deltaq1_32 = _mm_sub_epi32(sq2_32, deltaq1_32);
		deltaq1_32 = _mm_add_epi32(deltaq1_32, avg_32);
		deltaq1_32 = _mm_srai_epi32(deltaq1_32, 1);
		deltaq1_32 = _mm_max_epi32(deltaq1_32, _mm_set1_epi32(-c0));
		deltaq1_32 = _mm_min_epi32(deltaq1_32, _mm_set1_epi32(c0));

		//sp0[i] = min(max(sp0[i] + delta, 0), _peak);
		//sq0[i] = min(max(sq0[i] - delta, 0), _peak);

		//avx2にmaskstoreがあるが、16bitはない・・・ので全部ストアする
		temp = _mm_and_si128(mask, delta_32);
		sp0_32 = _mm_add_epi32(sp0_32, temp);
		sp0_32 = _mm_max_epi32(sp0_32, _mm_setzero_si128());
		sp0_32 = _mm_min_epi32(sp0_32, _mm_set1_epi32(_peak));
		sp0_32 = _mm_packus_epi32(sp0_32, _mm_setzero_si128());	//sse4.1

		sq0_32 = _mm_sub_epi32(sq0_32, temp);
		sq0_32 = _mm_max_epi32(sq0_32, _mm_setzero_si128());
		sq0_32 = _mm_min_epi32(sq0_32, _mm_set1_epi32(_peak));
		sq0_32 = _mm_packus_epi32(sq0_32, _mm_setzero_si128());	//sse4.1

		//if (ap < beta)
		//	sp1[i] += deltap1;
		//if (aq < beta)
		//	sq1[i] += deltaq1;

		temp = _mm_and_si128(mask4, deltap1_32);	//ifの
		temp = _mm_and_si128(mask, temp);			//分岐に入るかどうか
		sp1_32 = _mm_add_epi32(sp1_32, temp);
		sp1_32 = _mm_packus_epi32(sp1_32, _mm_setzero_si128());

		temp = _mm_and_si128(mask5, deltaq1_32);	//ifの
		temp = _mm_and_si128(mask, temp);			//分岐に入るかどうか
		sq1_32 = _mm_add_epi32(sq1_32, temp);
		sq1_32 = _mm_packus_epi32(sq1_32, _mm_setzero_si128());

		if (mode == HorProc) {	//水平モード
			_mm_storel_epi64((__m128i*)sp1, sp1_32);
			_mm_storel_epi64((__m128i*)sp0, sp0_32);
			_mm_storel_epi64((__m128i*)sq0, sq0_32);
			_mm_storel_epi64((__m128i*)sq1, sq1_32);
		}
		else {		//垂直モードの時は転置処理
			transpose_16bit_4x4(sp1_32, sp0_32, sq0_32, sq1_32, sp1_32, sp0_32, sq0_32, sq1_32);
			_mm_storel_epi64((__m128i*)(sp1 + 1), sp1_32);
			_mm_storel_epi64((__m128i*)(sp0 + 1), sp0_32);	//処理中のsp0はdstp[-3]で、出力先はdstp[-2]から
			_mm_storel_epi64((__m128i*)(sq0 + 1), sq0_32);
			_mm_storel_epi64((__m128i*)(sq1 + 1), sq1_32);
		}
	}

}


inline void deblockEdgeOPT8(uint8_t* VS_RESTRICT dstp, const unsigned stride, int mode) {
	const int alpha = _alpha;
	const int beta = _beta;
	const int c0 = _c0;
	const int c1 = _c1;

	const uint8_t* sp2;
	uint8_t* VS_RESTRICT sp1;
	uint8_t* VS_RESTRICT sp0;
	uint8_t* VS_RESTRICT sq0;
	uint8_t* VS_RESTRICT sq1;
	uint8_t* VS_RESTRICT sq2;
	uint8_t* VS_RESTRICT sq3;
	uint8_t* VS_RESTRICT sq4;
	uint8_t* VS_RESTRICT sq5;

	__m128i sp0_16, sp1_16, sq0_16, sq1_16, sp2_16, sq2_16;

	if (mode == HorProc) {	//水平モード
		sp2 = dstp - stride * 3;
		sp1 = dstp - stride * 2;
		sp0 = dstp - stride;
		sq0 = dstp;
		sq1 = dstp + stride;
		sq2 = dstp + stride * 2;

		sp2_16 = _mm_loadu_si64((__m128i*)sp2);	//本来はifの後でよい
		sp1_16 = _mm_loadu_si64((__m128i*)sp1);	//8個分
		sp0_16 = _mm_loadu_si64((__m128i*)sp0);
		sq0_16 = _mm_loadu_si64((__m128i*)sq0);
		sq1_16 = _mm_loadu_si64((__m128i*)sq1);
		sq2_16 = _mm_loadu_si64((__m128i*)sq2);	//本来はifの後でよい
	}
	else {		//垂直モードの時は転置処理
		sp1 = dstp - 3;
		sp0 = dstp - 3 + stride;
		sq0 = dstp - 3 + stride * 2;
		sq1 = dstp - 3 + stride * 3;
		sq2 = dstp - 3 + stride * 4;
		sq3 = dstp - 3 + stride * 5;
		sq4 = dstp - 3 + stride * 6;
		sq5 = dstp - 3 + stride * 7;

		auto data0 = _mm_loadu_si64((__m128i*)sp1);	//読み込みは8byte分(使うのは6byte)
		auto data1 = _mm_loadu_si64((__m128i*)sp0);
		auto data2 = _mm_loadu_si64((__m128i*)sq0);
		auto data3 = _mm_loadu_si64((__m128i*)sq1);
		auto data4 = _mm_loadu_si64((__m128i*)sq2);
		auto data5 = _mm_loadu_si64((__m128i*)sq3);
		auto data6 = _mm_loadu_si64((__m128i*)sq4);
		auto data7 = _mm_loadu_si64((__m128i*)sq5);

		transpose_8bit_6x8(data0, data1, data2, data3, data4, data5, data6, data7,
			sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);
	}

	Deblock::deblockEdgeOPT_cal8_sse4(sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);


	if (mode == HorProc) {	//水平モード
		_mm_storel_epi64((__m128i*)sp1, sp1_16);
		_mm_storel_epi64((__m128i*)sp0, sp0_16);
		_mm_storel_epi64((__m128i*)sq0, sq0_16);
		_mm_storel_epi64((__m128i*)sq1, sq1_16);
	}
	else {		//垂直モードの時は転置処理
		int out0, out1, out2, out3, out4, out5, out6, out7;
		//transpose_8bit_4x4x2(sp1_16, sp0_16, sq0_16, sq1_16,out0, out1, out2, out3);	//4x4x2でやろうと思ったけど1つ目の結果(q1)がないと2つ目の計算ができないことに気が付いた・・・orz

		transpose_8bit_8x4(sp1_16, sp0_16, sq0_16, sq1_16, out0, out1, out2, out3, out4, out5, out6, out7);
		*(uint32_t*)(sp1 + 1) = out0;	//dstp - 3 + 1 = dstp-2
		*(uint32_t*)(sp0 + 1) = out1;  //処理中のsp0はdstp[-3]で、出力先はdstp[-2]から
		*(uint32_t*)(sq0 + 1) = out2;
		*(uint32_t*)(sq1 + 1) = out3;
		*(uint32_t*)(sq2 + 1) = out4;
		*(uint32_t*)(sq3 + 1) = out5;
		*(uint32_t*)(sq4 + 1) = out6;
		*(uint32_t*)(sq5 + 1) = out7;
	}
}

inline void deblockEdgeOPT8(uint16_t* VS_RESTRICT dstp, const unsigned stride, int mode) {

	const uint16_t* sp2;
	uint16_t* VS_RESTRICT sp1;
	uint16_t* VS_RESTRICT sp0;
	uint16_t* VS_RESTRICT sq0;
	uint16_t* VS_RESTRICT sq1;
	uint16_t* VS_RESTRICT sq2;
	uint16_t* VS_RESTRICT sq3;
	uint16_t* VS_RESTRICT sq4;
	uint16_t* VS_RESTRICT sq5;

	__m128i sp0_16, sp1_16, sq0_16, sq1_16, sp2_16, sq2_16;

	if (mode == HorProc) {	//水平モード
		sp2 = dstp - stride * 3;
		sp1 = dstp - stride * 2;
		sp0 = dstp - stride;
		sq0 = dstp;
		sq1 = dstp + stride;
		sq2 = dstp + stride * 2;

		sp2_16 = _mm_loadu_si128((__m128i*)sp2);
		sp1_16 = _mm_loadu_si128((__m128i*)sp1);
		sp0_16 = _mm_loadu_si128((__m128i*)sp0);
		sq0_16 = _mm_loadu_si128((__m128i*)sq0);
		sq1_16 = _mm_loadu_si128((__m128i*)sq1);
		sq2_16 = _mm_loadu_si128((__m128i*)sq2);
	}
	else {
		sp1 = dstp - 3;
		sp0 = dstp - 3 + stride;
		sq0 = dstp - 3 + stride * 2;
		sq1 = dstp - 3 + stride * 3;
		sq2 = dstp - 3 + stride * 4;
		sq3 = dstp - 3 + stride * 5;
		sq4 = dstp - 3 + stride * 6;
		sq5 = dstp - 3 + stride * 7;

		auto data0 = _mm_loadu_si128((__m128i*)sp1);	//読み込みは8個分(使うのは6個)
		auto data1 = _mm_loadu_si128((__m128i*)sp0);
		auto data2 = _mm_loadu_si128((__m128i*)sq0);
		auto data3 = _mm_loadu_si128((__m128i*)sq1);
		auto data4 = _mm_loadu_si128((__m128i*)sq2);
		auto data5 = _mm_loadu_si128((__m128i*)sq3);
		auto data6 = _mm_loadu_si128((__m128i*)sq4);
		auto data7 = _mm_loadu_si128((__m128i*)sq5);

		transpose_16bit_6x8(data0, data1, data2, data3, data4, data5, data6, data7,
			sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);

	}

	Deblock::deblockEdgeOPT_cal16_AVX2(sp2_16, sp1_16, sp0_16, sq0_16, sq1_16, sq2_16);


	if (mode == HorProc) {	//水平モード
		_mm_storeu_si128((__m128i*)sp1, sp1_16);
		_mm_storeu_si128((__m128i*)sp0, sp0_16);
		_mm_storeu_si128((__m128i*)sq0, sq0_16);
		_mm_storeu_si128((__m128i*)sq1, sq1_16);
	}
	else {		//垂直モードの時は転置処理
		__m128i out0, out1, out2, out3, out4, out5, out6, out7;
		transpose_16bit_8x4(sp1_16, sp0_16, sq0_16, sq1_16, out0, out1, out2, out3, out4, out5, out6, out7);
		_mm_storel_epi64((__m128i*)(sp1 + 1), out0);	//dstp - 3 + 1 = dstp-2
		_mm_storel_epi64((__m128i*)(sp0 + 1), out1);
		_mm_storel_epi64((__m128i*)(sq0 + 1), out2);
		_mm_storel_epi64((__m128i*)(sq1 + 1), out3);
		_mm_storel_epi64((__m128i*)(sq2 + 1), out4);
		_mm_storel_epi64((__m128i*)(sq3 + 1), out5);
		_mm_storel_epi64((__m128i*)(sq4 + 1), out6);
		_mm_storel_epi64((__m128i*)(sq5 + 1), out7);
	}


}

inline void deblockEdgeOPT_cal16_AVX2(__m128i& sp2_16, __m128i& sp1_16, __m128i& sp0_16, __m128i& sq0_16, __m128i& sq1_16, __m128i& sq2_16)
{
	const int alpha = _alpha;
	const int beta = _beta;
	const int c0 = _c0;
	const int c1 = _c1;

	auto sp2_32 = _mm256_cvtepu16_epi32(sp2_16);
	auto sp1_32 = _mm256_cvtepu16_epi32(sp1_16);
	auto sp0_32 = _mm256_cvtepu16_epi32(sp0_16);	//AVX2
	auto sq0_32 = _mm256_cvtepu16_epi32(sq0_16);
	auto sq1_32 = _mm256_cvtepu16_epi32(sq1_16);
	auto sq2_32 = _mm256_cvtepu16_epi32(sq2_16);

	//for (unsigned i = 0; i < 4; i++) {
		//if (std::abs(sp0[i] - sq0[i]) < alpha && std::abs(sp1[i] - sp0[i]) < beta && std::abs(sq0[i] - sq1[i]) < beta) {
	auto tsub = _mm256_sub_epi32(sp0_32, sq0_32);
	auto tabs = _mm256_abs_epi32(tsub);			//
	auto mask1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(alpha), tabs);	//AVX2ではgt命令しかないみたい

	tsub = _mm256_sub_epi32(sp1_32, sp0_32);
	tabs = _mm256_abs_epi32(tsub);
	auto mask2 = _mm256_cmpgt_epi32(_mm256_set1_epi32(beta), tabs);

	tsub = _mm256_sub_epi32(sq0_32, sq1_32);
	tabs = _mm256_abs_epi32(tsub);
	auto mask3 = _mm256_cmpgt_epi32(_mm256_set1_epi32(beta), tabs);

	auto mask = _mm256_and_si256(mask1, mask2);
	mask = _mm256_and_si256(mask, mask3);	//ifはFF

	//if (~_mm_test_all_zeros(mask, mask))	//あってもなくてもあまり速度は変わらない？
	{

		//	const int ap = std::abs(sp2[i] - sp0[i]);
		//	const int aq = std::abs(sq2[i] - sq0[i]);


		tsub = _mm256_sub_epi32(sp2_32, sp0_32);
		auto ap_32 = _mm256_abs_epi32(tsub);

		tsub = _mm256_sub_epi32(sq2_32, sq0_32);
		auto aq_32 = _mm256_abs_epi32(tsub);


		//int c = c0;
		//if (ap < beta)
		//	c += c1;
		//if (aq < beta)
		//	c += c1;

		auto c_32 = _mm256_set1_epi32(c0);
		auto c1_32 = _mm256_set1_epi32(c1);
		auto mask4 = _mm256_cmpgt_epi32(_mm256_set1_epi32(beta), ap_32);
		auto temp = _mm256_and_si256(c1_32, mask4);
		c_32 = _mm256_add_epi32(c_32, temp);

		auto mask5 = _mm256_cmpgt_epi32(_mm256_set1_epi32(beta), aq_32);
		temp = _mm256_and_si256(c1_32, mask5);	//mask4と5の順番注意・・・元のcではqが先になっているが、逆にしておく↑のcも逆にした
		c_32 = _mm256_add_epi32(c_32, temp);
		auto c_32m = _mm256_sub_epi32(_mm256_setzero_si256(), c_32);

		//const int avg = (sp0[i] + sq0[i] + 1) >> 1;
		//const int delta = min(max(((sq0[i] - sp0[i]) * 4 + sp1[i] - sq1[i] + 4) >> 3, -c), c);
		//const int deltap1 = min(max((sp2[i] + avg - sp1[i] * 2) >> 1, -c0), c0);
		//const int deltaq1 = min(max((sq2[i] + avg - sq1[i] * 2) >> 1, -c0), c0);

		auto avg_32 = _mm256_add_epi32(sp0_32, sq0_32);
		avg_32 = _mm256_add_epi32(avg_32, _mm256_set1_epi32(1));
		avg_32 = _mm256_srai_epi32(avg_32, 1);

		auto delta_32 = _mm256_sub_epi32(sq0_32, sp0_32);
		delta_32 = _mm256_slli_epi32(delta_32, 2);
		delta_32 = _mm256_add_epi32(delta_32, sp1_32);
		delta_32 = _mm256_sub_epi32(delta_32, sq1_32);
		delta_32 = _mm256_add_epi32(delta_32, _mm256_set1_epi32(4));
		delta_32 = _mm256_srai_epi32(delta_32, 3);
		delta_32 = _mm256_max_epi32(delta_32, c_32m);
		delta_32 = _mm256_min_epi32(delta_32, c_32);

		auto deltap1_32 = _mm256_slli_epi32(sp1_32, 1);
		deltap1_32 = _mm256_sub_epi32(sp2_32, deltap1_32);
		deltap1_32 = _mm256_add_epi32(deltap1_32, avg_32);
		deltap1_32 = _mm256_srai_epi32(deltap1_32, 1);
		deltap1_32 = _mm256_max_epi32(deltap1_32, _mm256_set1_epi32(-c0));
		deltap1_32 = _mm256_min_epi32(deltap1_32, _mm256_set1_epi32(c0));

		auto deltaq1_32 = _mm256_slli_epi32(sq1_32, 1);
		deltaq1_32 = _mm256_sub_epi32(sq2_32, deltaq1_32);
		deltaq1_32 = _mm256_add_epi32(deltaq1_32, avg_32);
		deltaq1_32 = _mm256_srai_epi32(deltaq1_32, 1);
		deltaq1_32 = _mm256_max_epi32(deltaq1_32, _mm256_set1_epi32(-c0));
		deltaq1_32 = _mm256_min_epi32(deltaq1_32, _mm256_set1_epi32(c0));

		//sp0[i] = min(max(sp0[i] + delta, 0), _peak);
		//sq0[i] = min(max(sq0[i] - delta, 0), _peak);

		//avx2にmaskstoreがあるが、16bitはない・・・ので全部ストアする
		temp = _mm256_and_si256(mask, delta_32);
		sp0_32 = _mm256_add_epi32(sp0_32, temp);
		sp0_32 = _mm256_max_epi32(sp0_32, _mm256_setzero_si256());
		sp0_32 = _mm256_min_epi32(sp0_32, _mm256_set1_epi32(_peak));
		sp0_32 = _mm256_packus_epi32(sp0_32, _mm256_permute2x128_si256(sp0_32, sp0_32, 1));
		sp0_16 = _mm256_extracti128_si256(sp0_32, 0);

		sq0_32 = _mm256_sub_epi32(sq0_32, temp);
		sq0_32 = _mm256_max_epi32(sq0_32, _mm256_setzero_si256());
		sq0_32 = _mm256_min_epi32(sq0_32, _mm256_set1_epi32(_peak));
		sq0_32 = _mm256_packus_epi32(sq0_32, _mm256_permute2x128_si256(sq0_32, sq0_32, 1));
		sq0_16 = _mm256_extracti128_si256(sq0_32, 0);

		//if (ap < beta)
		//	sp1[i] += deltap1;
		//if (aq < beta)
		//	sq1[i] += deltaq1;

		temp = _mm256_and_si256(mask4, deltap1_32);	//ifの
		temp = _mm256_and_si256(mask, temp);			//分岐に入るかどうか
		sp1_32 = _mm256_add_epi32(sp1_32, temp);
		sp1_32 = _mm256_packus_epi32(sp1_32, _mm256_permute2x128_si256(sp1_32, sp1_32, 1));
		sp1_16 = _mm256_extracti128_si256(sp1_32, 0);

		temp = _mm256_and_si256(mask5, deltaq1_32);	//ifの
		temp = _mm256_and_si256(mask, temp);			//分岐に入るかどうか
		sq1_32 = _mm256_add_epi32(sq1_32, temp);
		sq1_32 = _mm256_packus_epi32(sq1_32, _mm256_permute2x128_si256(sq1_32, sq1_32, 1));
		sq1_16 = _mm256_extracti128_si256(sq1_32, 0);

	}
}
