//
// Created by longxiang on 3/10/23.
//

#ifndef LONGXIANG_DIST_UTILS_H
#define LONGXIANG_DIST_UTILS_H

#include "dist_header.h"
//#include <immintrin.h>


namespace utils{
    template<typename T>
    static T L2SqrNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        T *pVec1 = (T*) pVec1v;
        T *pVec2 = (T*) pVec2v;
        std::size_t dim = *((std::size_t *) dim_ptr);

        T diff, res=0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            diff = pVec1[idx] - pVec2[idx];
            res += diff * diff;
        }
        return res;
    }

    template<typename T>
    static float IPNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        T *pVec1 = (T*) pVec1v;
        T *pVec2 = (T*) pVec2v;
        std::size_t dim = *((std::size_t *) dim_ptr);

        T res=0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            res += pVec1[idx] * pVec2[idx];
        }
        return res;
    }

    template<typename T>
    static T NormSqr(const void *pVec, const void *dim_ptr) {
        T *pV = (T *) pVec;
        std::size_t dim =  *((std::size_t *) dim_ptr);

        T res = 0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            res += pV[idx] * pV[idx];
        }
        return res;
    }

#define _MM_SHFFLE(fp3, fp2, fp1, fp0) (((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | (fp0))

// Adapted from https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
static inline __m128 MaskedReadFloat(const std::size_t dim, const float* data) {
    assert(0<=dim && dim < 4 );
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (dim) {
        case 3:
            buf[2] = data[2];
        case 2:
            buf[1] = data[1];
        case 1:
            buf[0] = data[0];
    }
    return _mm_load_ps(buf);
}

static inline __m128i MaskedReadInt(const std::size_t dim, const int* data) {
    assert(0<=dim && dim < 4 );
    ALIGNED(16) int buf[4] = {0, 0, 0, 0};
    switch (dim) {
        case 3:
            buf[2] = data[2];
        case 2:
            buf[1] = data[1];
        case 1:
            buf[0] = data[0];
    }
    return _mm_load_si128((__m128i *)buf);
}

// Adapted from https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av
static float HsumFloat128(__m128 x) {
//    __m128 h64 = _mm_unpackhi_ps(x, x);
    __m128 h64 = _mm_shuffle_ps(x, x, _MM_SHFFLE(1, 0, 3, 2));
    __m128 sum64 = _mm_add_ps(h64, x);
    __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHFFLE(0, 1, 2, 3));
    __m128 sum32 = _mm_add_ps(sum64, h32);
    return _mm_cvtss_f32(sum32);
}

static int HsumInt128(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
//    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHFFLE(1, 0, 3, 2));
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHFFLE(1, 0, 3, 2));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

// Adapted from https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
//static float L2SqrFloatAVX512(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
static float L2SqrFloatAVX512(const float *pVec1, const float *pVec2, const std::size_t *dim_ptr) {
//    float *pVec1 = (float *) pVec1v;
//    float *pVec2 = (float *) pVec2v;
//    std::size_t dim = *((std::size_t *) dim_ptr);

    __m512 mx512, my512, diff512;
    __m512 sum512 = _mm512_setzero_ps();

//    while (dim >= 16) {
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;
//    }
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
//        dim -= 16;

    __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512), _mm512_extractf32x8_ps(sum512, 1));

//    if (dim >= 8) {
//        __m256 mx256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
//        __m256 my256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
//        __m256 diff256 = _mm256_sub_ps(mx256, my256);
//        sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
//        dim -= 8;
//    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 mx128, my128, diff128;

//    if (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
//        dim -= 4;
//    }

//    if (dim > 0) {
//        mx128 = MaskedReadFloat(dim, pVec1);
//        my128 = MaskedReadFloat(dim, pVec2);
//        diff128 = _mm_sub_ps(mx128, my128);
//        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
//    }
//    return HsumFloat128(sum128);
        __m128 h64 = _mm_shuffle_ps(sum128, sum128, _MM_SHFFLE(1, 0, 3, 2));
        __m128 sum64 = _mm_add_ps(h64, sum128);
        __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHFFLE(0, 1, 2, 3));
        __m128 sum32 = _mm_add_ps(sum64, h32);
        return _mm_cvtss_f32(sum32);
}
//    FAISS_PRAGMA_IMPRECISE_FUNCTION_END
static float L2SqrFloatAVX(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 sum256 = _mm256_setzero_ps();

    while (dim >= 8) {
        __m256 mx256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 my256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        __m256 diff256 = _mm256_sub_ps(mx256, my256);
        sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 mx128, my128, diff128;

    if (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        mx128 = MaskedReadFloat(dim, pVec1);
        my128 = MaskedReadFloat(dim, pVec2);
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
    }
    return HsumFloat128(sum128);
}

static float L2SqrFloatSSE(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 sum128 = _mm_setzero_ps();
    __m128 mx128, my128, diff128;

    while (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        mx128 = MaskedReadFloat(dim, pVec1);
        my128 = MaskedReadFloat(dim, pVec2);
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
    }
    return HsumFloat128(sum128);
}

static float InnerProductFloatAVX512(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m512 x512, y512, diff512;
    __m512 sum512 = _mm512_setzero_ps();

    while (dim >= 16) {
        x512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        y512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        sum512 = _mm512_fmadd_ps(x512, y512, sum512);
        dim -= 16;
    }
    __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512), _mm512_extractf32x8_ps(sum512, 1));

    if (dim>=8){
        __m256 x256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 y256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        sum256 = _mm256_fmadd_ps(x256, y256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 x128, y128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}

static float InnerProductFloatAVX(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 sum256 = _mm256_setzero_ps();

    while (dim>=8){
        __m256 x256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 y256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        sum256 = _mm256_fmadd_ps(x256, y256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 x128, y128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}

static float InnerProductFloatSSE(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 sum128 = _mm_setzero_ps();
    __m128 x128, y128;

    while (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}

static float NormSqrtFloatAVX512(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);
    __m512 x512;
    __m512 res512 = _mm512_setzero_ps();

    while(dim >= 16) {
        x512 = _mm512_loadu_ps(pV); pV += 16;
        res512 = _mm512_fmadd_ps(x512, x512, res512);
        dim -= 16;
    }
    __m256 res256 = _mm256_add_ps(_mm512_castps512_ps256(res512), _mm512_extractf32x8_ps(res512, 1));

    if (dim >= 8) {
        __m256 x256 = _mm256_loadu_ps(pV); pV += 8;
        res256 = _mm256_fmadd_ps(x256, x256, res256);
        dim -= 8;
    }
    __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256), _mm256_extractf128_ps(res256, 1));
    __m128 x128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}

static float NormSqrtFloatAVX(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 res256 = _mm256_setzero_ps();
    while (dim >= 8) {
        __m256 x256 = _mm256_loadu_ps(pV); pV += 8;
        res256 = _mm256_fmadd_ps(x256, x256, res256);
        dim -= 8;
    }
    __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256), _mm256_extractf128_ps(res256, 1));
    __m128 x128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}

static float NormSqrtFloatSSE(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 res128 = _mm_setzero_ps();
    __m128 x128;

    while (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}

static float L2SqrtWithNormAVX512(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
}

static float L2SqrtWithNormAVX(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
}

static float L2SqrtWithNormSSE(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
}

} // namespace utils

#endif //LONGXIANG_DIST_UTILS_H