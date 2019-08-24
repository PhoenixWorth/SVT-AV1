/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <immintrin.h>
#include "aom_dsp_rtcd.h"
#include "convolve.h"
#include "convolve_avx2.h"
#include "EbDefinitions.h"
#include "EbMemory_SSE4_1.h"

#define DIST_WTD_CONVOLVE_VERTICAL_FILTER_2TAP                                       \
    __m256i s[6];                                                                    \
                                                                                     \
    for (i = 0; i < h; i += 2) {                                                     \
        const int16_t *data = &t_block[i * im_stride];                               \
                                                                                     \
        const __m256i s4 =                                                           \
            _mm256_loadu_si256((__m256i *)(data + 0 * im_stride));                   \
        const __m256i s5 =                                                           \
            _mm256_loadu_si256((__m256i *)(data + 1 * im_stride));                   \
                                                                                     \
        s[0] = _mm256_unpacklo_epi16(s4, s5);                                        \
        s[1] = _mm256_unpackhi_epi16(s4, s5);                                        \
                                                                                     \
        const __m256i res_a = convolve16_2tap_avx2(s, coeffs_v);                     \
        const __m256i res_a_round = _mm256_sra_epi32(                                \
            _mm256_add_epi32(res_a, round_const_v), round_shift_v);                  \
                                                                                     \
        if (w - j > 4) {                                                             \
            const __m256i res_b = convolve16_2tap_avx2(s + 1, coeffs_v);             \
            const __m256i res_b_round = _mm256_sra_epi32(                            \
                _mm256_add_epi32(res_b, round_const_v), round_shift_v);              \
            const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_b_round);    \
            const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                                     \
            if (do_average) {                                                        \
                const __m256i data_ref_0 =                                           \
                    load_line2_avx2(&dst[i * dst_stride + j],                        \
                        &dst[i * dst_stride + j + dst_stride]);                      \
                const __m256i comp_avg_res = comp_avg(&data_ref_0, &res_unsigned,    \
                    &wt, use_jnt_comp_avg);                                          \
                                                                                     \
                const __m256i round_result = convolve_rounding(                      \
                    &comp_avg_res, &offset_const, &rounding_const, rounding_shift);  \
                                                                                     \
                const __m256i res_8 =                                                \
                    _mm256_packus_epi16(round_result, round_result);                 \
                const __m128i res_0 = _mm256_castsi256_si128(res_8);                 \
                const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);            \
                                                                                     \
                _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);    \
                _mm_storel_epi64(                                                    \
                    (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1); \
            }                                                                        \
            else {                                                                   \
                const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);          \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);       \
                                                                                     \
                const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);     \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),  \
                    res_1);                                                          \
            }                                                                        \
        }                                                                            \
        else {                                                                       \
            const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_a_round);    \
            const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                                     \
            if (do_average) {                                                        \
                const __m256i data_ref_0 =                                           \
                    load_line2_avx2(&dst[i * dst_stride + j],                        \
                        &dst[i * dst_stride + j + dst_stride]);                      \
                                                                                     \
                const __m256i comp_avg_res = comp_avg(&data_ref_0, &res_unsigned,    \
                    &wt, use_jnt_comp_avg);                                          \
                                                                                     \
                const __m256i round_result = convolve_rounding(                      \
                    &comp_avg_res, &offset_const, &rounding_const, rounding_shift);  \
                                                                                     \
                const __m256i res_8 =                                                \
                    _mm256_packus_epi16(round_result, round_result);                 \
                const __m128i res_0 = _mm256_castsi256_si128(res_8);                 \
                const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);            \
                                                                                     \
                *(uint32_t *)(&dst0[i * dst_stride0 + j]) =                          \
                    _mm_cvtsi128_si32(res_0);                                        \
                *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =            \
                    _mm_cvtsi128_si32(res_1);                                        \
                                                                                     \
            }                                                                        \
            else {                                                                   \
                const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);          \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);       \
                                                                                     \
                const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);     \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),  \
                    res_1);                                                          \
            }                                                                        \
        }                                                                            \
    }

#define DIST_WTD_CONVOLVE_VERTICAL_FILTER_4TAP                                       \
    __m256i s[6];                                                                    \
    __m256i s0 = _mm256_loadu_si256((__m256i *)(t_block + 0 * im_stride));           \
    __m256i s1 = _mm256_loadu_si256((__m256i *)(t_block + 1 * im_stride));           \
    __m256i s2 = _mm256_loadu_si256((__m256i *)(t_block + 2 * im_stride));           \
    __m256i s3 = _mm256_loadu_si256((__m256i *)(t_block + 3 * im_stride));           \
                                                                                     \
    s[0] = _mm256_unpacklo_epi16(s0, s1);                                            \
    s[1] = _mm256_unpacklo_epi16(s2, s3);                                            \
                                                                                     \
    s[3] = _mm256_unpackhi_epi16(s0, s1);                                            \
    s[4] = _mm256_unpackhi_epi16(s2, s3);                                            \
                                                                                     \
    for (i = 0; i < h; i += 2) {                                                     \
        const int16_t *data = &t_block[i * im_stride];                               \
                                                                                     \
        const __m256i s4 =                                                           \
            _mm256_loadu_si256((__m256i *)(data + 4 * im_stride));                   \
        const __m256i s5 =                                                           \
            _mm256_loadu_si256((__m256i *)(data + 5 * im_stride));                   \
                                                                                     \
        s[2] = _mm256_unpacklo_epi16(s4, s5);                                        \
        s[5] = _mm256_unpackhi_epi16(s4, s5);                                        \
                                                                                     \
        const __m256i res_a = convolve16_4tap_avx2(s, coeffs_v + 1);                 \
        const __m256i res_a_round = _mm256_sra_epi32(                                \
            _mm256_add_epi32(res_a, round_const_v), round_shift_v);                  \
                                                                                     \
        if (w - j > 4) {                                                             \
            const __m256i res_b = convolve16_4tap_avx2(s + 3, coeffs_v + 1);         \
            const __m256i res_b_round = _mm256_sra_epi32(                            \
                _mm256_add_epi32(res_b, round_const_v), round_shift_v);              \
            const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_b_round);    \
            const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                                     \
            if (do_average) {                                                        \
                const __m256i data_ref_0 =                                           \
                    load_line2_avx2(&dst[i * dst_stride + j],                        \
                        &dst[i * dst_stride + j + dst_stride]);                      \
                const __m256i comp_avg_res = comp_avg(&data_ref_0, &res_unsigned,    \
                    &wt, use_jnt_comp_avg);                                          \
                                                                                     \
                const __m256i round_result = convolve_rounding(                      \
                    &comp_avg_res, &offset_const, &rounding_const, rounding_shift);  \
                                                                                     \
                const __m256i res_8 =                                                \
                    _mm256_packus_epi16(round_result, round_result);                 \
                const __m128i res_0 = _mm256_castsi256_si128(res_8);                 \
                const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);            \
                                                                                     \
                _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);    \
                _mm_storel_epi64(                                                    \
                    (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1); \
            }                                                                        \
            else {                                                                   \
                const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);          \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);       \
                                                                                     \
                const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);     \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),  \
                    res_1);                                                          \
            }                                                                        \
        }                                                                            \
        else {                                                                       \
            const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_a_round);    \
            const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                                     \
            if (do_average) {                                                        \
                const __m256i data_ref_0 =                                           \
                    load_line2_avx2(&dst[i * dst_stride + j],                        \
                        &dst[i * dst_stride + j + dst_stride]);                      \
                                                                                     \
                const __m256i comp_avg_res = comp_avg(&data_ref_0, &res_unsigned,    \
                    &wt, use_jnt_comp_avg);                                          \
                                                                                     \
                const __m256i round_result = convolve_rounding(                      \
                    &comp_avg_res, &offset_const, &rounding_const, rounding_shift);  \
                                                                                     \
                const __m256i res_8 =                                                \
                    _mm256_packus_epi16(round_result, round_result);                 \
                const __m128i res_0 = _mm256_castsi256_si128(res_8);                 \
                const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);            \
                                                                                     \
                *(uint32_t *)(&dst0[i * dst_stride0 + j]) =                          \
                    _mm_cvtsi128_si32(res_0);                                        \
                *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =            \
                    _mm_cvtsi128_si32(res_1);                                        \
                                                                                     \
            }                                                                        \
            else {                                                                   \
                const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);          \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);       \
                                                                                     \
                const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);     \
                _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),  \
                    res_1);                                                          \
            }                                                                        \
        }                                                                            \
        s[0] = s[1];                                                                 \
        s[1] = s[2];                                                                 \
        s[3] = s[4];                                                                 \
        s[4] = s[5];                                                                 \
    }

#define DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP                                 \
  __m256i s[8];                                                                \
  __m256i s0 = _mm256_loadu_si256((__m256i *)(t_block + 0 * im_stride));       \
  __m256i s1 = _mm256_loadu_si256((__m256i *)(t_block + 1 * im_stride));       \
  __m256i s2 = _mm256_loadu_si256((__m256i *)(t_block + 2 * im_stride));       \
  __m256i s3 = _mm256_loadu_si256((__m256i *)(t_block + 3 * im_stride));       \
  __m256i s4 = _mm256_loadu_si256((__m256i *)(t_block + 4 * im_stride));       \
  __m256i s5 = _mm256_loadu_si256((__m256i *)(t_block + 5 * im_stride));       \
                                                                               \
  s[0] = _mm256_unpacklo_epi16(s0, s1);                                        \
  s[1] = _mm256_unpacklo_epi16(s2, s3);                                        \
  s[2] = _mm256_unpacklo_epi16(s4, s5);                                        \
                                                                               \
  s[4] = _mm256_unpackhi_epi16(s0, s1);                                        \
  s[5] = _mm256_unpackhi_epi16(s2, s3);                                        \
  s[6] = _mm256_unpackhi_epi16(s4, s5);                                        \
                                                                               \
  for (i = 0; i < h; i += 2) {                                                 \
    const int16_t *data = &t_block[i * im_stride];                             \
                                                                               \
    const __m256i s6 = _mm256_loadu_si256((__m256i *)(data + 6 * im_stride));  \
    const __m256i s7 = _mm256_loadu_si256((__m256i *)(data + 7 * im_stride));  \
                                                                               \
    s[3] = _mm256_unpacklo_epi16(s6, s7);                                      \
    s[7] = _mm256_unpackhi_epi16(s6, s7);                                      \
                                                                               \
    const __m256i res_a = convolve16_8tap_avx2(s, coeffs_v);                   \
    const __m256i res_a_round = _mm256_sra_epi32(                              \
        _mm256_add_epi32(res_a, round_const_v), round_shift_v);                \
                                                                               \
    if (w - j > 4) {                                                           \
      const __m256i res_b = convolve16_8tap_avx2(s + 4, coeffs_v);             \
      const __m256i res_b_round = _mm256_sra_epi32(                            \
          _mm256_add_epi32(res_b, round_const_v), round_shift_v);              \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_b_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);      \
        _mm_storel_epi64(                                                      \
            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);   \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);         \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),    \
                        res_1);                                                \
      }                                                                        \
    } else {                                                                   \
      const __m256i res_16b = _mm256_packs_epi32(res_a_round, res_a_round);    \
      const __m256i res_unsigned = _mm256_add_epi16(res_16b, offset_const);    \
                                                                               \
      if (do_average) {                                                        \
        const __m256i data_ref_0 = load_line2_avx2(                            \
            &dst[i * dst_stride + j], &dst[i * dst_stride + j + dst_stride]);  \
                                                                               \
        const __m256i comp_avg_res =                                           \
            comp_avg(&data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);       \
                                                                               \
        const __m256i round_result = convolve_rounding(                        \
            &comp_avg_res, &offset_const, &rounding_const, rounding_shift);    \
                                                                               \
        const __m256i res_8 = _mm256_packus_epi16(round_result, round_result); \
        const __m128i res_0 = _mm256_castsi256_si128(res_8);                   \
        const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);              \
                                                                               \
        *(uint32_t *)(&dst0[i * dst_stride0 + j]) = _mm_cvtsi128_si32(res_0);  \
        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =              \
            _mm_cvtsi128_si32(res_1);                                          \
                                                                               \
      } else {                                                                 \
        const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);            \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);         \
                                                                               \
        const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);       \
        _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),    \
                        res_1);                                                \
      }                                                                        \
    }                                                                          \
                                                                               \
    s[0] = s[1];                                                               \
    s[1] = s[2];                                                               \
    s[2] = s[3];                                                               \
                                                                               \
    s[4] = s[5];                                                               \
    s[5] = s[6];                                                               \
    s[6] = s[7];                                                               \
  }

static INLINE __m256i unpack_weights_avx2(ConvolveParams *conv_params) {
    const int32_t w0 = conv_params->fwd_offset;
    const int32_t w1 = conv_params->bck_offset;
    const __m256i wt0 = _mm256_set1_epi16(w0);
    const __m256i wt1 = _mm256_set1_epi16(w1);
    const __m256i wt = _mm256_unpacklo_epi16(wt0, wt1);
    return wt;
}

static INLINE __m256i load_line2_avx2(const void *a, const void *b) {
    return _mm256_permute2x128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)a)),
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i *)b)), 0x20);
}

void eb_av1_jnt_convolve_x_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst0, int32_t dst_stride0, int32_t w, int32_t h,
    InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn, const int32_t subpel_y_qn,
    ConvolveParams *conv_params) {
    ConvBufType *dst = conv_params->dst;
    int32_t dst_stride = conv_params->dst_stride;
    const int32_t bd = 8;
    int32_t i, j;
    const int32_t bits = FILTER_BITS - conv_params->round_1;
    const __m256i wt = unpack_weights_avx2(conv_params);
    const int32_t do_average = conv_params->do_average;
    const int32_t use_jnt_comp_avg = conv_params->use_jnt_comp_avg;
    const int32_t offset_0 =
        bd + 2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const int32_t offset = (1 << offset_0) + (1 << (offset_0 - 1));
    const __m256i offset_const = _mm256_set1_epi16(offset);
    const int32_t rounding_shift =
        2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const __m256i rounding_const = _mm256_set1_epi16((1 << rounding_shift) >> 1);

    assert(bits >= 0);
    assert(conv_params->round_0 > 0);

    const __m256i round_const =
        _mm256_set1_epi16((1 << (conv_params->round_0 - 1)) >> 1);
    const __m128i round_shift = _mm_cvtsi32_si128(conv_params->round_0 - 1);

    (void)filter_params_y;
    (void)subpel_y_qn;

    __m256i filt[4], coeffs[4];

    filt[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
    filt[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);

    if (is_convolve_2tap(filter_params_x->filter_ptr)) {
        const int32_t fo_horiz = 0;
        const uint8_t *const src_ptr = src - fo_horiz;

        prepare_half_coeffs_2tap_avx2(filter_params_x, subpel_x_qn, coeffs);

        for (i = 0; i < h; i += 2) {
            const uint8_t *src_data = src_ptr + i * src_stride;
            ConvBufType *dst_data = dst + i * dst_stride;
            for (j = 0; j < w; j += 8) {
                const __m256i data =
                    load_line2_avx2(&src_data[j], &src_data[j + src_stride]);

                __m256i res = convolve_x_2tap_avx2(data, coeffs, filt);
                res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const), round_shift);
                res = _mm256_slli_epi16(res, bits);

                const __m256i res_unsigned = _mm256_add_epi16(res, offset_const);

                // Accumulate values into the destination buffer
                if (do_average) {
                    const __m256i data_ref_0 =
                        load_line2_avx2(&dst_data[j], &dst_data[j + dst_stride]);
                    const __m256i comp_avg_res =
                        comp_avg(&data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);

                    const __m256i round_result = convolve_rounding(
                        &comp_avg_res, &offset_const, &rounding_const, rounding_shift);

                    const __m256i res_8 = _mm256_packus_epi16(round_result, round_result);
                    const __m128i res_0 = _mm256_castsi256_si128(res_8);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);

                    if (w > 4) {
                        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);
                        _mm_storel_epi64(
                            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);
                    }
                    else {
                        *(uint32_t *)(&dst0[i * dst_stride0 + j]) =
                            _mm_cvtsi128_si32(res_0);
                        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =
                            _mm_cvtsi128_si32(res_1);
                    }
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);

                    const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),
                        res_1);
                }
            }
        }
    }
    else if (is_convolve_4tap(filter_params_x->filter_ptr)) {
        const int32_t fo_horiz = 1;
        const uint8_t *const src_ptr = src - fo_horiz;

        prepare_half_coeffs_8tap_avx2(filter_params_x, subpel_x_qn, coeffs);

        for (i = 0; i < h; i += 2) {
            const uint8_t *src_data = src_ptr + i * src_stride;
            ConvBufType *dst_data = dst + i * dst_stride;
            for (j = 0; j < w; j += 8) {
                const __m256i data =
                    load_line2_avx2(&src_data[j], &src_data[j + src_stride]);

                __m256i res = convolve_x_4tap_avx2(data, coeffs + 1, filt);
                res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const), round_shift);
                res = _mm256_slli_epi16(res, bits);

                const __m256i res_unsigned = _mm256_add_epi16(res, offset_const);

                // Accumulate values into the destination buffer
                if (do_average) {
                    const __m256i data_ref_0 =
                        load_line2_avx2(&dst_data[j], &dst_data[j + dst_stride]);
                    const __m256i comp_avg_res =
                        comp_avg(&data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);

                    const __m256i round_result = convolve_rounding(
                        &comp_avg_res, &offset_const, &rounding_const, rounding_shift);

                    const __m256i res_8 = _mm256_packus_epi16(round_result, round_result);
                    const __m128i res_0 = _mm256_castsi256_si128(res_8);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);

                    if (w > 4) {
                        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);
                        _mm_storel_epi64(
                            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);
                    }
                    else {
                        *(uint32_t *)(&dst0[i * dst_stride0 + j]) =
                            _mm_cvtsi128_si32(res_0);
                        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =
                            _mm_cvtsi128_si32(res_1);
                    }
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);

                    const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),
                        res_1);
                }
            }
        }
    }
    else {
        const int32_t fo_horiz = filter_params_x->taps / 2 - 1;
        const uint8_t *const src_ptr = src - fo_horiz;

        prepare_half_coeffs_8tap_avx2(filter_params_x, subpel_x_qn, coeffs);
        filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
        filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);
        for (i = 0; i < h; i += 2) {
            const uint8_t *src_data = src_ptr + i * src_stride;
            ConvBufType *dst_data = dst + i * dst_stride;
            for (j = 0; j < w; j += 8) {
                const __m256i data =
                    load_line2_avx2(&src_data[j], &src_data[j + src_stride]);

                __m256i res = convolve_x_8tap_avx2(data, coeffs, filt);

                res = _mm256_sra_epi16(_mm256_add_epi16(res, round_const), round_shift);

                res = _mm256_slli_epi16(res, bits);

                const __m256i res_unsigned = _mm256_add_epi16(res, offset_const);

                // Accumulate values into the destination buffer
                if (do_average) {
                    const __m256i data_ref_0 =
                        load_line2_avx2(&dst_data[j], &dst_data[j + dst_stride]);
                    const __m256i comp_avg_res =
                        comp_avg(&data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);

                    const __m256i round_result = convolve_rounding(
                        &comp_avg_res, &offset_const, &rounding_const, rounding_shift);

                    const __m256i res_8 = _mm256_packus_epi16(round_result, round_result);
                    const __m128i res_0 = _mm256_castsi256_si128(res_8);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);

                    if (w > 4) {
                        _mm_storel_epi64((__m128i *)(&dst0[i * dst_stride0 + j]), res_0);
                        _mm_storel_epi64(
                            (__m128i *)((&dst0[i * dst_stride0 + j + dst_stride0])), res_1);
                    }
                    else {
                        *(uint32_t *)(&dst0[i * dst_stride0 + j]) =
                            _mm_cvtsi128_si32(res_0);
                        *(uint32_t *)(&dst0[i * dst_stride0 + j + dst_stride0]) =
                            _mm_cvtsi128_si32(res_1);
                    }
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j]), res_0);

                    const __m128i res_1 = _mm256_extracti128_si256(res_unsigned, 1);
                    _mm_store_si128((__m128i *)(&dst[i * dst_stride + j + dst_stride]),
                        res_1);
                }
            }
        }
    }
}

// =============================================================================

SIMD_INLINE void jnt_convolve_y_comp_avg_2tap_32_avx2(
    const uint8_t *const src, const __m256i *const coeffs, const __m256i factor,
    const __m256i offset, const __m256i s0, __m256i *const s1,
    ConvBufType *const dst, uint8_t *const dst8) {
    __m256i r[2];

    convolve_y_2tap_32_kernel_avx2(src, coeffs, s0, s1, r);
    jnt_comp_avg_32_avx2(r, factor, offset, dst, dst8);
}

SIMD_INLINE void jnt_convolve_y_avg_2tap_32_avx2(
    const uint8_t *const src, const __m256i *const coeffs, const __m256i offset,
    const __m256i s0, __m256i *const s1, const ConvBufType *const dst,
    uint8_t *const dst8) {
    __m256i r[2];

    convolve_y_2tap_32_kernel_avx2(src, coeffs, s0, s1, r);
    jnt_avg_32_avx2(r, offset, dst, dst8);
}

SIMD_INLINE void jnt_convolve_y_no_avg_2tap_32_avx2(
    const uint8_t *const src, const __m256i *const coeffs, const __m256i offset,
    const __m256i s0, __m256i *const s1, ConvBufType *const dst) {
    __m256i r[2];

    convolve_y_2tap_32_kernel_avx2(src, coeffs, s0, s1, r);
    jnt_no_avg_32_avx2(r, offset, dst);
}

void eb_av1_jnt_convolve_y_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst8, int32_t dst8_stride, int32_t w,
    int32_t h, InterpFilterParams *filter_params_x,
    InterpFilterParams *filter_params_y,
    const int32_t subpel_x_q4,
    const int32_t subpel_y_q4,
    ConvolveParams *conv_params) {
    const __m128i offset_no_avg_128 = _mm_set1_epi16((1 << 12) + (1 << 11));
    const __m128i offset_avg_128 = _mm_set1_epi16((1 << 12) + (1 << 11) - 16);
    const __m128i offset_comp_avg_128 =
        _mm_set1_epi32(conv_params->bck_offset * ((1 << 12) + (1 << 11)) -
        (1 << 16) - (1 << 15) + 128);
    const __m256i offset_comp_avg_256 =
        _mm256_set1_epi32(conv_params->bck_offset * ((1 << 12) + (1 << 11)) -
        (1 << 16) - (1 << 15) + 128);
    const __m256i offset_no_avg_256 = _mm256_set1_epi16((1 << 12) + (1 << 11));
    const __m256i offset_avg_256 =
        _mm256_set1_epi16((1 << 12) + (1 << 11) - 16);
    const int32_t dst_stride = conv_params->dst_stride;
    ConvBufType *dst = conv_params->dst;
    int32_t x, y;
    __m128i factor_128;
    __m128i coeffs_128[4];
    __m256i factor_256;
    __m256i coeffs_256[4];

    (void)filter_params_x;
    (void)subpel_x_q4;
    (void)conv_params;

    assert(conv_params->round_0 == 3);
    assert(conv_params->round_1 == COMPOUND_ROUND1_BITS);

    if (conv_params->use_jnt_comp_avg) {
        const int32_t factor =
            conv_params->fwd_offset | (conv_params->bck_offset << 16);
        factor_128 = _mm_set1_epi32(factor);
        factor_256 = _mm256_set1_epi32(factor);
    }

    if (is_convolve_2tap(filter_params_y->filter_ptr)) {
        // vert_filt as 2 tap
        const uint8_t *src_ptr = src;

        y = h;

        if (w <= 4) {
            prepare_half_coeffs_2tap_ssse3(
                filter_params_y, subpel_y_q4, coeffs_128);

            if (w == 2) {
                __m128i s_16[2];

                s_16[0] = _mm_cvtsi32_si128(*(int16_t *)src_ptr);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m128i res = convolve_y_2tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16);
                            jnt_comp_avg_2x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m128i res = convolve_y_2tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16);
                            jnt_avg_2x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m128i res = convolve_y_2tap_2x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_16);
                        jnt_no_avg_2x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m128i s_32[2];

                assert(w == 4);

                s_32[0] = _mm_cvtsi32_si128(*(int32_t *)src_ptr);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m128i res = convolve_y_2tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32);
                            jnt_comp_avg_4x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m128i res = convolve_y_2tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32);
                            jnt_avg_4x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m128i res = convolve_y_2tap_4x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_32);
                        jnt_no_avg_4x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
        else {
            prepare_half_coeffs_2tap_avx2(
                filter_params_y, subpel_y_q4, coeffs_256);

            if (w == 8) {
                __m128i s_64[2];

                s_64[0] = _mm_loadl_epi64((__m128i *)src_ptr);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m256i res = convolve_y_2tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64);
                            jnt_comp_avg_8x2_avx2(res,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m256i res = convolve_y_2tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64);
                            jnt_avg_8x2_sse2(res,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m256i res = convolve_y_2tap_8x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_64);
                        jnt_no_avg_8x2_avx2(
                            res, offset_no_avg_256, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else if (w == 16) {
                __m128i s_128[2];
                __m256i r[2];

                s_128[0] = _mm_loadu_si128((__m128i *)src_ptr);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            convolve_y_2tap_16x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_128, r);
                            jnt_comp_avg_16x2_avx2(r,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            convolve_y_2tap_16x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_128, r);
                            jnt_avg_16x2_sse2(r,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        convolve_y_2tap_16x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_128, r);
                        jnt_no_avg_16x2_avx2(
                            r, offset_no_avg_256, dst, dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else if (w == 32) {
                __m256i s_256[2];

                s_256[0] = _mm256_loadu_si256((__m256i *)src_ptr);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0],
                                &s_256[1],
                                dst,
                                dst8);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1],
                                &s_256[0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0],
                                &s_256[1],
                                dst,
                                dst8);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1],
                                &s_256[0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        jnt_convolve_y_no_avg_2tap_32_avx2(src_ptr + src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0],
                            &s_256[1],
                            dst);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1],
                            &s_256[0],
                            dst + dst_stride);
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else if (w == 64) {
                __m256i s_256[2][2];

                s_256[0][0] = _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                s_256[0][1] = _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][0],
                                &s_256[1][0],
                                dst,
                                dst8);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride + 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][1],
                                &s_256[1][1],
                                dst + 32,
                                dst8 + 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][0],
                                &s_256[0][0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][1],
                                &s_256[0][1],
                                dst + dst_stride + 32,
                                dst8 + dst8_stride + 32);

                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][0],
                                &s_256[1][0],
                                dst,
                                dst8);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride + 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][1],
                                &s_256[1][1],
                                dst + 32,
                                dst8 + 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][0],
                                &s_256[0][0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][1],
                                &s_256[0][1],
                                dst + dst_stride + 32,
                                dst8 + dst8_stride + 32);

                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        jnt_convolve_y_no_avg_2tap_32_avx2(src_ptr + src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][0],
                            &s_256[1][0],
                            dst);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + src_stride + 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][1],
                            &s_256[1][1],
                            dst + 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][0],
                            &s_256[0][0],
                            dst + dst_stride);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride + 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][1],
                            &s_256[0][1],
                            dst + dst_stride + 32);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m256i s_256[2][4];

                assert(w == 128);

                s_256[0][0] = _mm256_loadu_si256((__m256i *)(src_ptr + 0 * 32));
                s_256[0][1] = _mm256_loadu_si256((__m256i *)(src_ptr + 1 * 32));
                s_256[0][2] = _mm256_loadu_si256((__m256i *)(src_ptr + 2 * 32));
                s_256[0][3] = _mm256_loadu_si256((__m256i *)(src_ptr + 3 * 32));

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][0],
                                &s_256[1][0],
                                dst,
                                dst8);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride + 1 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][1],
                                &s_256[1][1],
                                dst + 1 * 32,
                                dst8 + 1 * 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride + 2 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][2],
                                &s_256[1][2],
                                dst + 2 * 32,
                                dst8 + 2 * 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + src_stride + 3 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[0][3],
                                &s_256[1][3],
                                dst + 3 * 32,
                                dst8 + 3 * 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][0],
                                &s_256[0][0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 1 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][1],
                                &s_256[0][1],
                                dst + dst_stride + 1 * 32,
                                dst8 + dst8_stride + 1 * 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 2 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][2],
                                &s_256[0][2],
                                dst + dst_stride + 2 * 32,
                                dst8 + dst8_stride + 2 * 32);
                            jnt_convolve_y_comp_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 3 * 32,
                                coeffs_256,
                                factor_256,
                                offset_comp_avg_256,
                                s_256[1][3],
                                &s_256[0][3],
                                dst + dst_stride + 3 * 32,
                                dst8 + dst8_stride + 3 * 32);

                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][0],
                                &s_256[1][0],
                                dst,
                                dst8);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride + 1 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][1],
                                &s_256[1][1],
                                dst + 1 * 32,
                                dst8 + 1 * 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride + 2 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][2],
                                &s_256[1][2],
                                dst + 2 * 32,
                                dst8 + 2 * 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + src_stride + 3 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[0][3],
                                &s_256[1][3],
                                dst + 3 * 32,
                                dst8 + 3 * 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][0],
                                &s_256[0][0],
                                dst + dst_stride,
                                dst8 + dst8_stride);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 1 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][1],
                                &s_256[0][1],
                                dst + dst_stride + 1 * 32,
                                dst8 + dst8_stride + 1 * 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 2 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][2],
                                &s_256[0][2],
                                dst + dst_stride + 2 * 32,
                                dst8 + dst8_stride + 2 * 32);
                            jnt_convolve_y_avg_2tap_32_avx2(
                                src_ptr + 2 * src_stride + 3 * 32,
                                coeffs_256,
                                offset_avg_256,
                                s_256[1][3],
                                &s_256[0][3],
                                dst + dst_stride + 3 * 32,
                                dst8 + dst8_stride + 3 * 32);

                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        jnt_convolve_y_no_avg_2tap_32_avx2(src_ptr + src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][0],
                            &s_256[1][0],
                            dst);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + src_stride + 1 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][1],
                            &s_256[1][1],
                            dst + 1 * 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + src_stride + 2 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][2],
                            &s_256[1][2],
                            dst + 2 * 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + src_stride + 3 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[0][3],
                            &s_256[1][3],
                            dst + 3 * 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][0],
                            &s_256[0][0],
                            dst + dst_stride);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride + 1 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][1],
                            &s_256[0][1],
                            dst + dst_stride + 1 * 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride + 2 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][2],
                            &s_256[0][2],
                            dst + dst_stride + 2 * 32);
                        jnt_convolve_y_no_avg_2tap_32_avx2(
                            src_ptr + 2 * src_stride + 3 * 32,
                            coeffs_256,
                            offset_no_avg_256,
                            s_256[1][3],
                            &s_256[0][3],
                            dst + dst_stride + 3 * 32);

                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
    }
    else if (is_convolve_4tap(filter_params_y->filter_ptr)) {
        // vert_filt as 4 tap
        const uint8_t *src_ptr = src - src_stride;

        y = h;

        if (w <= 4) {
            prepare_half_coeffs_4tap_ssse3(
                filter_params_y, subpel_y_q4, coeffs_128);

            if (w == 2) {
                __m128i s_16[4], ss_128[2];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_4tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_comp_avg_2x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_4tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_avg_2x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m128i res = convolve_y_4tap_2x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_16, ss_128);
                        jnt_no_avg_2x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }

            }
            else {
                __m128i s_32[4], ss_128[2];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_4tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_comp_avg_4x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_4tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_avg_4x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m128i res = convolve_y_4tap_4x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_32, ss_128);
                        jnt_no_avg_4x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
        else {
            prepare_half_coeffs_4tap_avx2(
                filter_params_y, subpel_y_q4, coeffs_256);

            if (w == 8) {
                __m128i s_64[4];
                __m256i ss_256[2];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m256i res = convolve_y_4tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_comp_avg_8x2_avx2(res,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m256i res = convolve_y_4tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_avg_8x2_sse2(res,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m256i res = convolve_y_4tap_8x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_64, ss_256);
                        jnt_no_avg_8x2_avx2(
                            res, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m128i s_128[4];
                __m256i ss_256[4], r[2];

                assert(w == 16);

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[2] = _mm256_unpackhi_epi8(src01, src12);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            convolve_y_4tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_comp_avg_16x2_avx2(r,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[2] = ss_256[3];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            convolve_y_4tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_avg_16x2_sse2(r,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[2] = ss_256[3];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        convolve_y_4tap_16x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_128, ss_256, r);
                        jnt_no_avg_16x2_avx2(
                            r, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        ss_256[2] = ss_256[3];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
    }
    else if (is_convolve_6tap(filter_params_y->filter_ptr)) {
        // vert_filt as 6 tap
        const uint8_t *src_ptr = src - 2 * src_stride;

        if (w <= 4) {
            prepare_half_coeffs_6tap_ssse3(
                filter_params_y, subpel_y_q4, coeffs_128);

            y = h;

            if (w == 2) {
                __m128i s_16[6], ss_128[3];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));
                s_16[3] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 3 * src_stride));
                s_16[4] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 4 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);
                const __m128i src23 = _mm_unpacklo_epi16(s_16[2], s_16[3]);
                const __m128i src34 = _mm_unpacklo_epi16(s_16[3], s_16[4]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_6tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_comp_avg_2x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_6tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_avg_2x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m128i res = convolve_y_6tap_2x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_16, ss_128);
                        jnt_no_avg_2x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        ss_128[1] = ss_128[2];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m128i s_32[6], ss_128[3];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));
                s_32[3] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 3 * src_stride));
                s_32[4] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 4 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);
                const __m128i src23 = _mm_unpacklo_epi32(s_32[2], s_32[3]);
                const __m128i src34 = _mm_unpacklo_epi32(s_32[3], s_32[4]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_6tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_comp_avg_4x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m128i res = convolve_y_6tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_avg_4x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m128i res = convolve_y_6tap_4x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_32, ss_128);
                        jnt_no_avg_4x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        ss_128[1] = ss_128[2];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
        else {
            prepare_half_coeffs_6tap_avx2(
                filter_params_y, subpel_y_q4, coeffs_256);

            if (w == 8) {
                __m128i s_64[6];
                __m256i ss_256[3];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));
                s_64[3] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 3 * src_stride));
                s_64[4] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 4 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);
                const __m256i src23 = _mm256_setr_m128i(s_64[2], s_64[3]);
                const __m256i src34 = _mm256_setr_m128i(s_64[3], s_64[4]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);

                y = h;

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m256i res = convolve_y_6tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_comp_avg_8x2_avx2(res,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            const __m256i res = convolve_y_6tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_avg_8x2_sse2(res,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        const __m256i res = convolve_y_6tap_8x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_64, ss_256);
                        jnt_no_avg_8x2_avx2(
                            res, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else if (w == 16) {
                __m128i s_128[6];
                __m256i ss_256[6], r[2];

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                s_128[3] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 3 * src_stride));
                s_128[4] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 4 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);
                const __m256i src23 = _mm256_setr_m128i(s_128[2], s_128[3]);
                const __m256i src34 = _mm256_setr_m128i(s_128[3], s_128[4]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);

                ss_256[3] = _mm256_unpackhi_epi8(src01, src12);
                ss_256[4] = _mm256_unpackhi_epi8(src23, src34);

                y = h;

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            src_ptr += 2 * src_stride;
                            convolve_y_6tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_comp_avg_16x2_avx2(r,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[3] = ss_256[4];
                            ss_256[4] = ss_256[5];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            src_ptr += 2 * src_stride;
                            convolve_y_6tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_avg_16x2_sse2(r,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[3] = ss_256[4];
                            ss_256[4] = ss_256[5];
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        src_ptr += 2 * src_stride;
                        convolve_y_6tap_16x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_128, ss_256, r);
                        jnt_no_avg_16x2_avx2(
                            r, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        ss_256[3] = ss_256[4];
                        ss_256[4] = ss_256[5];
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m256i s_256[6], ss_256[6], tt_256[6], r[4];

                assert(!(w % 32));

                x = 0;
                do {
                    const uint8_t *s = src_ptr + x;
                    ConvBufType *d = dst + x;
                    uint8_t *d8 = dst8 + x;

                    s_256[0] =
                        _mm256_loadu_si256((__m256i *)(s + 0 * src_stride));
                    s_256[1] =
                        _mm256_loadu_si256((__m256i *)(s + 1 * src_stride));
                    s_256[2] =
                        _mm256_loadu_si256((__m256i *)(s + 2 * src_stride));
                    s_256[3] =
                        _mm256_loadu_si256((__m256i *)(s + 3 * src_stride));
                    s_256[4] =
                        _mm256_loadu_si256((__m256i *)(s + 4 * src_stride));

                    ss_256[0] = _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                    ss_256[1] = _mm256_unpacklo_epi8(s_256[2], s_256[3]);
                    ss_256[3] = _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                    ss_256[4] = _mm256_unpackhi_epi8(s_256[2], s_256[3]);

                    tt_256[0] = _mm256_unpacklo_epi8(s_256[1], s_256[2]);
                    tt_256[1] = _mm256_unpacklo_epi8(s_256[3], s_256[4]);
                    tt_256[3] = _mm256_unpackhi_epi8(s_256[1], s_256[2]);
                    tt_256[4] = _mm256_unpackhi_epi8(s_256[3], s_256[4]);

                    y = h;

                    if (conv_params->do_average) {
                        if (conv_params->use_jnt_comp_avg) {
                            do {
                                s += 2 * src_stride;
                                convolve_y_6tap_32x2_avx2(s,
                                    src_stride,
                                    coeffs_256,
                                    s_256,
                                    ss_256,
                                    tt_256,
                                    r);
                                jnt_comp_avg_32_avx2(
                                    r, factor_256, offset_comp_avg_256, d, d8);
                                jnt_comp_avg_32_avx2(r + 2,
                                    factor_256,
                                    offset_comp_avg_256,
                                    d + dst_stride,
                                    d8 + dst8_stride);

                                ss_256[0] = ss_256[1];
                                ss_256[1] = ss_256[2];
                                ss_256[3] = ss_256[4];
                                ss_256[4] = ss_256[5];

                                tt_256[0] = tt_256[1];
                                tt_256[1] = tt_256[2];
                                tt_256[3] = tt_256[4];
                                tt_256[4] = tt_256[5];
                                d += 2 * dst_stride;
                                d8 += 2 * dst8_stride;
                                y -= 2;
                            } while (y);
                        }
                        else {
                            do {
                                s += 2 * src_stride;
                                convolve_y_6tap_32x2_avx2(s,
                                    src_stride,
                                    coeffs_256,
                                    s_256,
                                    ss_256,
                                    tt_256,
                                    r);
                                jnt_avg_32_avx2(r, offset_avg_256, d, d8);
                                jnt_avg_32_avx2(r + 2,
                                    offset_avg_256,
                                    d + dst_stride,
                                    d8 + dst8_stride);

                                ss_256[0] = ss_256[1];
                                ss_256[1] = ss_256[2];
                                ss_256[3] = ss_256[4];
                                ss_256[4] = ss_256[5];

                                tt_256[0] = tt_256[1];
                                tt_256[1] = tt_256[2];
                                tt_256[3] = tt_256[4];
                                tt_256[4] = tt_256[5];
                                d += 2 * dst_stride;
                                d8 += 2 * dst8_stride;
                                y -= 2;
                            } while (y);
                        }
                    }
                    else {
                        do {
                            s += 2 * src_stride;
                            convolve_y_6tap_32x2_avx2(s,
                                src_stride,
                                coeffs_256,
                                s_256,
                                ss_256,
                                tt_256,
                                r);
                            jnt_no_avg_32_avx2(r, offset_no_avg_256, d);
                            jnt_no_avg_32_avx2(
                                r + 2, offset_no_avg_256, d + dst_stride);

                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[3] = ss_256[4];
                            ss_256[4] = ss_256[5];

                            tt_256[0] = tt_256[1];
                            tt_256[1] = tt_256[2];
                            tt_256[3] = tt_256[4];
                            tt_256[4] = tt_256[5];
                            d += 2 * dst_stride;
                            y -= 2;
                        } while (y);
                    }

                    x += 32;
                } while (x < w);
            }
        }
    }
    else {
        // vert_filt as 8 tap
        const uint8_t *src_ptr = src - 3 * src_stride;

        if (w <= 4) {
            prepare_half_coeffs_8tap_ssse3(
                filter_params_y, subpel_y_q4, coeffs_128);

            y = h;

            if (w == 2) {
                __m128i s_16[8], ss_128[4];

                s_16[0] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 0 * src_stride));
                s_16[1] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 1 * src_stride));
                s_16[2] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 2 * src_stride));
                s_16[3] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 3 * src_stride));
                s_16[4] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 4 * src_stride));
                s_16[5] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 5 * src_stride));
                s_16[6] =
                    _mm_cvtsi32_si128(*(int16_t *)(src_ptr + 6 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi16(s_16[0], s_16[1]);
                const __m128i src12 = _mm_unpacklo_epi16(s_16[1], s_16[2]);
                const __m128i src23 = _mm_unpacklo_epi16(s_16[2], s_16[3]);
                const __m128i src34 = _mm_unpacklo_epi16(s_16[3], s_16[4]);
                const __m128i src45 = _mm_unpacklo_epi16(s_16[4], s_16[5]);
                const __m128i src56 = _mm_unpacklo_epi16(s_16[5], s_16[6]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);
                ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m128i res = convolve_y_8tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_comp_avg_2x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            ss_128[2] = ss_128[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m128i res = convolve_y_8tap_2x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_16, ss_128);
                            jnt_avg_2x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            ss_128[2] = ss_128[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m128i res = convolve_y_8tap_2x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_16, ss_128);
                        jnt_no_avg_2x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        ss_128[1] = ss_128[2];
                        ss_128[2] = ss_128[3];
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m128i s_32[8], ss_128[4];

                assert(w == 4);

                s_32[0] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 0 * src_stride));
                s_32[1] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 1 * src_stride));
                s_32[2] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 2 * src_stride));
                s_32[3] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 3 * src_stride));
                s_32[4] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 4 * src_stride));
                s_32[5] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 5 * src_stride));
                s_32[6] =
                    _mm_cvtsi32_si128(*(int32_t *)(src_ptr + 6 * src_stride));

                const __m128i src01 = _mm_unpacklo_epi32(s_32[0], s_32[1]);
                const __m128i src12 = _mm_unpacklo_epi32(s_32[1], s_32[2]);
                const __m128i src23 = _mm_unpacklo_epi32(s_32[2], s_32[3]);
                const __m128i src34 = _mm_unpacklo_epi32(s_32[3], s_32[4]);
                const __m128i src45 = _mm_unpacklo_epi32(s_32[4], s_32[5]);
                const __m128i src56 = _mm_unpacklo_epi32(s_32[5], s_32[6]);

                ss_128[0] = _mm_unpacklo_epi8(src01, src12);
                ss_128[1] = _mm_unpacklo_epi8(src23, src34);
                ss_128[2] = _mm_unpacklo_epi8(src45, src56);

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m128i res = convolve_y_8tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_comp_avg_4x2_sse2(res,
                                factor_128,
                                offset_comp_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            ss_128[2] = ss_128[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m128i res = convolve_y_8tap_4x2_ssse3(
                                src_ptr, src_stride, coeffs_128, s_32, ss_128);
                            jnt_avg_4x2_sse2(res,
                                offset_avg_128,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_128[0] = ss_128[1];
                            ss_128[1] = ss_128[2];
                            ss_128[2] = ss_128[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m128i res = convolve_y_8tap_4x2_ssse3(
                            src_ptr, src_stride, coeffs_128, s_32, ss_128);
                        jnt_no_avg_4x2_sse2(
                            res, offset_no_avg_128, dst, dst_stride);
                        ss_128[0] = ss_128[1];
                        ss_128[1] = ss_128[2];
                        ss_128[2] = ss_128[3];
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
        }
        else {
            prepare_half_coeffs_8tap_avx2(
                filter_params_y, subpel_y_q4, coeffs_256);

            if (w == 8) {
                __m128i s_64[8];
                __m256i ss_256[4];

                s_64[0] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 0 * src_stride));
                s_64[1] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 1 * src_stride));
                s_64[2] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 2 * src_stride));
                s_64[3] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 3 * src_stride));
                s_64[4] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 4 * src_stride));
                s_64[5] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 5 * src_stride));
                s_64[6] =
                    _mm_loadl_epi64((__m128i *)(src_ptr + 6 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_64[0], s_64[1]);
                const __m256i src12 = _mm256_setr_m128i(s_64[1], s_64[2]);
                const __m256i src23 = _mm256_setr_m128i(s_64[2], s_64[3]);
                const __m256i src34 = _mm256_setr_m128i(s_64[3], s_64[4]);
                const __m256i src45 = _mm256_setr_m128i(s_64[4], s_64[5]);
                const __m256i src56 = _mm256_setr_m128i(s_64[5], s_64[6]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);
                ss_256[2] = _mm256_unpacklo_epi8(src45, src56);

                y = h;

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            const __m256i res = convolve_y_8tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_comp_avg_8x2_avx2(res,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[2] = ss_256[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            const __m256i res = convolve_y_8tap_8x2_avx2(
                                src_ptr, src_stride, coeffs_256, s_64, ss_256);
                            jnt_avg_8x2_sse2(res,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[2] = ss_256[3];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        const __m256i res = convolve_y_8tap_8x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_64, ss_256);
                        jnt_no_avg_8x2_avx2(
                            res, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        ss_256[2] = ss_256[3];
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else if (w == 16) {
                __m128i s_128[8];
                __m256i ss_256[8], r[2];

                s_128[0] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 0 * src_stride));
                s_128[1] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 1 * src_stride));
                s_128[2] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 2 * src_stride));
                s_128[3] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 3 * src_stride));
                s_128[4] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 4 * src_stride));
                s_128[5] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 5 * src_stride));
                s_128[6] =
                    _mm_loadu_si128((__m128i *)(src_ptr + 6 * src_stride));

                // Load lines a and b. Line a to lower 128, line b to upper
                // 128
                const __m256i src01 = _mm256_setr_m128i(s_128[0], s_128[1]);
                const __m256i src12 = _mm256_setr_m128i(s_128[1], s_128[2]);
                const __m256i src23 = _mm256_setr_m128i(s_128[2], s_128[3]);
                const __m256i src34 = _mm256_setr_m128i(s_128[3], s_128[4]);
                const __m256i src45 = _mm256_setr_m128i(s_128[4], s_128[5]);
                const __m256i src56 = _mm256_setr_m128i(s_128[5], s_128[6]);

                ss_256[0] = _mm256_unpacklo_epi8(src01, src12);
                ss_256[1] = _mm256_unpacklo_epi8(src23, src34);
                ss_256[2] = _mm256_unpacklo_epi8(src45, src56);

                ss_256[4] = _mm256_unpackhi_epi8(src01, src12);
                ss_256[5] = _mm256_unpackhi_epi8(src23, src34);
                ss_256[6] = _mm256_unpackhi_epi8(src45, src56);

                y = h;

                if (conv_params->do_average) {
                    if (conv_params->use_jnt_comp_avg) {
                        do {
                            convolve_y_8tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_comp_avg_16x2_avx2(r,
                                factor_256,
                                offset_comp_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[2] = ss_256[3];
                            ss_256[4] = ss_256[5];
                            ss_256[5] = ss_256[6];
                            ss_256[6] = ss_256[7];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                    else {
                        do {
                            convolve_y_8tap_16x2_avx2(src_ptr,
                                src_stride,
                                coeffs_256,
                                s_128,
                                ss_256,
                                r);
                            jnt_avg_16x2_sse2(r,
                                offset_avg_256,
                                dst,
                                dst_stride,
                                dst8,
                                dst8_stride);
                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[2] = ss_256[3];
                            ss_256[4] = ss_256[5];
                            ss_256[5] = ss_256[6];
                            ss_256[6] = ss_256[7];
                            src_ptr += 2 * src_stride;
                            dst += 2 * dst_stride;
                            dst8 += 2 * dst8_stride;
                            y -= 2;
                        } while (y);
                    }
                }
                else {
                    do {
                        convolve_y_8tap_16x2_avx2(
                            src_ptr, src_stride, coeffs_256, s_128, ss_256, r);
                        jnt_no_avg_16x2_avx2(
                            r, offset_no_avg_256, dst, dst_stride);
                        ss_256[0] = ss_256[1];
                        ss_256[1] = ss_256[2];
                        ss_256[2] = ss_256[3];
                        ss_256[4] = ss_256[5];
                        ss_256[5] = ss_256[6];
                        ss_256[6] = ss_256[7];
                        src_ptr += 2 * src_stride;
                        dst += 2 * dst_stride;
                        y -= 2;
                    } while (y);
                }
            }
            else {
                __m256i s_256[8], ss_256[8], tt_256[8], r[4];

                assert(!(w % 32));

                x = 0;
                do {
                    const uint8_t *s = src_ptr + x;
                    ConvBufType *d = dst + x;
                    uint8_t *d8 = dst8 + x;

                    s_256[0] =
                        _mm256_loadu_si256((__m256i *)(s + 0 * src_stride));
                    s_256[1] =
                        _mm256_loadu_si256((__m256i *)(s + 1 * src_stride));
                    s_256[2] =
                        _mm256_loadu_si256((__m256i *)(s + 2 * src_stride));
                    s_256[3] =
                        _mm256_loadu_si256((__m256i *)(s + 3 * src_stride));
                    s_256[4] =
                        _mm256_loadu_si256((__m256i *)(s + 4 * src_stride));
                    s_256[5] =
                        _mm256_loadu_si256((__m256i *)(s + 5 * src_stride));
                    s_256[6] =
                        _mm256_loadu_si256((__m256i *)(s + 6 * src_stride));

                    ss_256[0] = _mm256_unpacklo_epi8(s_256[0], s_256[1]);
                    ss_256[1] = _mm256_unpacklo_epi8(s_256[2], s_256[3]);
                    ss_256[2] = _mm256_unpacklo_epi8(s_256[4], s_256[5]);
                    ss_256[4] = _mm256_unpackhi_epi8(s_256[0], s_256[1]);
                    ss_256[5] = _mm256_unpackhi_epi8(s_256[2], s_256[3]);
                    ss_256[6] = _mm256_unpackhi_epi8(s_256[4], s_256[5]);

                    tt_256[0] = _mm256_unpacklo_epi8(s_256[1], s_256[2]);
                    tt_256[1] = _mm256_unpacklo_epi8(s_256[3], s_256[4]);
                    tt_256[2] = _mm256_unpacklo_epi8(s_256[5], s_256[6]);
                    tt_256[4] = _mm256_unpackhi_epi8(s_256[1], s_256[2]);
                    tt_256[5] = _mm256_unpackhi_epi8(s_256[3], s_256[4]);
                    tt_256[6] = _mm256_unpackhi_epi8(s_256[5], s_256[6]);

                    y = h;

                    if (conv_params->do_average) {
                        if (conv_params->use_jnt_comp_avg) {
                            do {
                                convolve_y_8tap_32x2_avx2(s,
                                    src_stride,
                                    coeffs_256,
                                    s_256,
                                    ss_256,
                                    tt_256,
                                    r);
                                jnt_comp_avg_32_avx2(
                                    r, factor_256, offset_comp_avg_256, d, d8);
                                jnt_comp_avg_32_avx2(r + 2,
                                    factor_256,
                                    offset_comp_avg_256,
                                    d + dst_stride,
                                    d8 + dst8_stride);

                                ss_256[0] = ss_256[1];
                                ss_256[1] = ss_256[2];
                                ss_256[2] = ss_256[3];
                                ss_256[4] = ss_256[5];
                                ss_256[5] = ss_256[6];
                                ss_256[6] = ss_256[7];

                                tt_256[0] = tt_256[1];
                                tt_256[1] = tt_256[2];
                                tt_256[2] = tt_256[3];
                                tt_256[4] = tt_256[5];
                                tt_256[5] = tt_256[6];
                                tt_256[6] = tt_256[7];
                                s += 2 * src_stride;
                                d += 2 * dst_stride;
                                d8 += 2 * dst8_stride;
                                y -= 2;
                            } while (y);
                        }
                        else {
                            do {
                                convolve_y_8tap_32x2_avx2(s,
                                    src_stride,
                                    coeffs_256,
                                    s_256,
                                    ss_256,
                                    tt_256,
                                    r);
                                jnt_avg_32_avx2(r, offset_avg_256, d, d8);
                                jnt_avg_32_avx2(r + 2,
                                    offset_avg_256,
                                    d + dst_stride,
                                    d8 + dst8_stride);

                                ss_256[0] = ss_256[1];
                                ss_256[1] = ss_256[2];
                                ss_256[2] = ss_256[3];
                                ss_256[4] = ss_256[5];
                                ss_256[5] = ss_256[6];
                                ss_256[6] = ss_256[7];

                                tt_256[0] = tt_256[1];
                                tt_256[1] = tt_256[2];
                                tt_256[2] = tt_256[3];
                                tt_256[4] = tt_256[5];
                                tt_256[5] = tt_256[6];
                                tt_256[6] = tt_256[7];
                                s += 2 * src_stride;
                                d += 2 * dst_stride;
                                d8 += 2 * dst8_stride;
                                y -= 2;
                            } while (y);
                        }
                    }
                    else {
                        do {
                            convolve_y_8tap_32x2_avx2(s,
                                src_stride,
                                coeffs_256,
                                s_256,
                                ss_256,
                                tt_256,
                                r);
                            jnt_no_avg_32_avx2(r, offset_no_avg_256, d);
                            jnt_no_avg_32_avx2(
                                r + 2, offset_no_avg_256, d + dst_stride);

                            ss_256[0] = ss_256[1];
                            ss_256[1] = ss_256[2];
                            ss_256[2] = ss_256[3];
                            ss_256[4] = ss_256[5];
                            ss_256[5] = ss_256[6];
                            ss_256[6] = ss_256[7];

                            tt_256[0] = tt_256[1];
                            tt_256[1] = tt_256[2];
                            tt_256[2] = tt_256[3];
                            tt_256[4] = tt_256[5];
                            tt_256[5] = tt_256[6];
                            tt_256[6] = tt_256[7];
                            s += 2 * src_stride;
                            d += 2 * dst_stride;
                            y -= 2;
                        } while (y);
                    }

                    x += 32;
                } while (x < w);
            }
        }
    }
}

// =============================================================================

void eb_av1_jnt_convolve_2d_avx2(const uint8_t *src, int32_t src_stride,
    uint8_t *dst0, int32_t dst_stride0, int32_t w,
    int32_t h, InterpFilterParams *filter_params_x,
    InterpFilterParams *filter_params_y,
    const int32_t subpel_x_qn,
    const int32_t subpel_y_qn,
    ConvolveParams *conv_params) {
    ConvBufType *dst = conv_params->dst;
    int32_t dst_stride = conv_params->dst_stride;
    const int32_t bd = 8;
    const int32_t h_tap = get_convolve_tap(filter_params_x->filter_ptr);
    const int32_t v_tap = get_convolve_tap(filter_params_y->filter_ptr);

    DECLARE_ALIGNED(32, int16_t, im_block[(MAX_SB_SIZE + MAX_FILTER_TAP) * 8]);

    int32_t im_stride = 8;
    int32_t i;
    const __m256i wt = unpack_weights_avx2(conv_params);
    const int32_t do_average = conv_params->do_average;
    const int32_t use_jnt_comp_avg = conv_params->use_jnt_comp_avg;
    const int32_t offset_0 =
        bd + 2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const int32_t offset = (1 << offset_0) + (1 << (offset_0 - 1));
    const __m256i offset_const = _mm256_set1_epi16(offset);
    const int32_t rounding_shift =
        2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const __m256i rounding_const =
        _mm256_set1_epi16((1 << rounding_shift) >> 1);

    assert(conv_params->round_0 > 0);

    const __m256i round_const_h =
        _mm256_set1_epi16(((1 << (conv_params->round_0 - 1)) >> 1) +
        (1 << (bd + FILTER_BITS - 2)));
    const __m128i round_shift_h = _mm_cvtsi32_si128(conv_params->round_0 - 1);

    const __m256i round_const_v = _mm256_set1_epi32(
        ((1 << conv_params->round_1) >> 1) -
        (1 << (bd + 2 * FILTER_BITS - conv_params->round_0 - 1)));
    const __m128i round_shift_v = _mm_cvtsi32_si128(conv_params->round_1);

    __m256i filt[4], coeffs_h[4], coeffs_v[4];

    filt[0] = _mm256_load_si256((__m256i const *)filt1_global_avx2);
    filt[1] = _mm256_load_si256((__m256i const *)filt2_global_avx2);

    prepare_half_coeffs_8tap_avx2(filter_params_x, subpel_x_qn, coeffs_h);
    prepare_coeffs_8tap_avx2(filter_params_y, subpel_y_qn, coeffs_v);

    if (h_tap == 2) {
        int32_t im_h = h + filter_params_y->taps - 1;
        const int32_t fo_vert = filter_params_y->taps / 2 - 1;
        const int32_t fo_horiz = 0;
        const uint8_t *const src_ptr = src - fo_vert * src_stride - fo_horiz;

        prepare_half_coeffs_2tap_avx2(filter_params_x, subpel_x_qn, coeffs_h);

        if (v_tap == 2) {
            const int16_t *const t_block = im_block + 3 * im_stride;
            prepare_coeffs_2tap_avx2(filter_params_y, subpel_y_qn, coeffs_v);
            for (int32_t j = 0; j < w; j += 8) {
                CONVOLVE_SR_HORIZONTAL_FILTER_2TAP;
                DIST_WTD_CONVOLVE_VERTICAL_FILTER_2TAP;
            }
        }
        else if (v_tap == 4) {
            const int16_t *const t_block = im_block + 2 * im_stride;
            for (int32_t j = 0; j < w; j += 8) {
                CONVOLVE_SR_HORIZONTAL_FILTER_2TAP;
                DIST_WTD_CONVOLVE_VERTICAL_FILTER_4TAP;
            }
        }
        else {
            const int16_t *const t_block = im_block + 0 * im_stride;
            for (int32_t j = 0; j < w; j += 8) {
                CONVOLVE_SR_HORIZONTAL_FILTER_2TAP;
                DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP;
            }
        }
    }
    else if (h_tap == 4) {
        int32_t im_h = h + filter_params_y->taps - 1;
        const int32_t fo_vert = filter_params_y->taps / 2 - 1;
        const int32_t fo_horiz = 1;
        const uint8_t *const src_ptr = src - fo_vert * src_stride - fo_horiz;
        const int16_t *const t_block = im_block;
        for (int32_t j = 0; j < w; j += 8) {
            CONVOLVE_SR_HORIZONTAL_FILTER_4TAP;
            DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP;
        }
    }
    else if (v_tap == 4) {
        int32_t im_h = h + 3;
        const int32_t fo_vert = 1;
        const int32_t fo_horiz = filter_params_x->taps / 2 - 1;
        const uint8_t *const src_ptr = src - fo_vert * src_stride - fo_horiz;
        const int16_t *const t_block = im_block;

        filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
        filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

        for (int32_t j = 0; j < w; j += 8) {
            CONVOLVE_SR_HORIZONTAL_FILTER_8TAP;
            DIST_WTD_CONVOLVE_VERTICAL_FILTER_4TAP;
        }
    }
    else {
        int32_t im_h = h + filter_params_y->taps - 1;
        const int32_t fo_vert = filter_params_y->taps / 2 - 1;
        const int32_t fo_horiz = filter_params_x->taps / 2 - 1;
        const uint8_t *const src_ptr = src - fo_vert * src_stride - fo_horiz;
        const int16_t *const t_block = im_block;

        filt[2] = _mm256_load_si256((__m256i const *)filt3_global_avx2);
        filt[3] = _mm256_load_si256((__m256i const *)filt4_global_avx2);

        for (int32_t j = 0; j < w; j += 8) {
            CONVOLVE_SR_HORIZONTAL_FILTER_8TAP;
            DIST_WTD_CONVOLVE_VERTICAL_FILTER_8TAP;
        }
    }
}

void eb_av1_jnt_convolve_2d_copy_avx2(
    const uint8_t *src, int32_t src_stride, uint8_t *dst0, int32_t dst_stride0,
    int32_t w, int32_t h, InterpFilterParams *filter_params_x,
    InterpFilterParams *filter_params_y, const int32_t subpel_x_q4,
    const int32_t subpel_y_q4, ConvolveParams *conv_params) {
    const int32_t bd = 8;
    ConvBufType *dst = conv_params->dst;
    int32_t dst_stride = conv_params->dst_stride;
    (void)filter_params_x;
    (void)filter_params_y;
    (void)subpel_x_q4;
    (void)subpel_y_q4;

    const int32_t bits =
        FILTER_BITS * 2 - conv_params->round_1 - conv_params->round_0;
    const __m128i left_shift = _mm_cvtsi32_si128(bits);
    const int32_t do_average = conv_params->do_average;
    const int32_t use_jnt_comp_avg = conv_params->use_jnt_comp_avg;
    const __m256i wt = unpack_weights_avx2(conv_params);
    const __m256i zero = _mm256_setzero_si256();

    const int32_t offset_0 =
        bd + 2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const int32_t offset = (1 << offset_0) + (1 << (offset_0 - 1));
    const __m256i offset_const = _mm256_set1_epi16(offset);
    const int32_t rounding_shift =
        2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const __m256i rounding_const =
        _mm256_set1_epi16((1 << rounding_shift) >> 1);
    int32_t i, j;

    if (!(w % 16)) {
        for (i = 0; i < h; i += 1) {
            for (j = 0; j < w; j += 16) {
                const __m256i src_16bit = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i *)(&src[i * src_stride + j])));

                const __m256i res = _mm256_sll_epi16(src_16bit, left_shift);
                const __m256i res_unsigned =
                    _mm256_add_epi16(res, offset_const);

                if (do_average) {
                    const __m256i data_ref_0 = _mm256_loadu_si256(
                        (__m256i *)(&dst[i * dst_stride + j]));

                    const __m256i comp_avg_res = comp_avg(
                        &data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);

                    const __m256i round_result =
                        convolve_rounding(&comp_avg_res,
                            &offset_const,
                            &rounding_const,
                            rounding_shift);

                    const __m256i res_8 =
                        _mm256_packus_epi16(round_result, round_result);
                    const __m256i res_0 = _mm256_permute4x64_epi64(res_8, 0xD8);

                    _mm_storeu_si128((__m128i *)(&dst0[i * dst_stride0 + j]),
                        _mm256_castsi256_si128(res_0));
                }
                else {
                    _mm256_storeu_si256((__m256i *)(&dst[i * dst_stride + j]),
                        res_unsigned);
                }
            }
        }
    }
    else if (!(w % 4)) {
        for (i = 0; i < h; i += 2) {
            for (j = 0; j < w; j += 8) {
                const __m128i src_row_0 =
                    _mm_loadl_epi64((__m128i *)(&src[i * src_stride + j]));
                const __m128i src_row_1 = _mm_loadl_epi64(
                    (__m128i *)(&src[i * src_stride + j + src_stride]));
                // since not all compilers yet support _mm256_set_m128i()
                const __m256i src_10 = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(src_row_0), src_row_1, 1);

                const __m256i src_16bit = _mm256_unpacklo_epi8(src_10, zero);

                const __m256i res = _mm256_sll_epi16(src_16bit, left_shift);

                const __m256i res_unsigned =
                    _mm256_add_epi16(res, offset_const);

                // Accumulate values into the destination buffer
                if (do_average) {
                    const __m256i data_ref_0 =
                        load_line2_avx2(&dst[i * dst_stride + j],
                            &dst[i * dst_stride + j + dst_stride]);

                    const __m256i comp_avg_res = comp_avg(
                        &data_ref_0, &res_unsigned, &wt, use_jnt_comp_avg);

                    const __m256i round_result =
                        convolve_rounding(&comp_avg_res,
                            &offset_const,
                            &rounding_const,
                            rounding_shift);

                    const __m256i res_8 =
                        _mm256_packus_epi16(round_result, round_result);
                    const __m128i res_0 = _mm256_castsi256_si128(res_8);
                    const __m128i res_1 = _mm256_extracti128_si256(res_8, 1);

                    if (w > 4) {
                        _mm_storel_epi64(
                            (__m128i *)(&dst0[i * dst_stride0 + j]), res_0);
                        _mm_storel_epi64(
                            (__m128i *)((
                                &dst0[i * dst_stride0 + j + dst_stride0])),
                            res_1);
                    }
                    else {
                        *(uint32_t *)(&dst0[i * dst_stride0 + j]) =
                            _mm_cvtsi128_si32(res_0);
                        *(uint32_t
                            *)(&dst0[i * dst_stride0 + j + dst_stride0]) =
                            _mm_cvtsi128_si32(res_1);
                    }
                }
                else {
                    const __m128i res_0 = _mm256_castsi256_si128(res_unsigned);
                    _mm_storeu_si128((__m128i *)(&dst[i * dst_stride + j]),
                        res_0);

                    const __m128i res_1 =
                        _mm256_extracti128_si256(res_unsigned, 1);
                    _mm_storeu_si128(
                        (__m128i *)(&dst[i * dst_stride + j + dst_stride]),
                        res_1);
                }
            }
        }
    }
}
