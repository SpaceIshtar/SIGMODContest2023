//
// Created by longxiang on 3/13/23.
//

#include "utils/dist_func.h"
#include "gtest/gtest.h"
#include <random>

namespace {
    TEST(DistTest, HsumFLoatTest) {
        ALIGNED(16) float buf[4] = {1, 3, 5, 7};
        __m128 data = _mm_load_ps(buf);
        float h_sum = utils::HsumFloat128(data);
        EXPECT_EQ(h_sum, 16);
    }

    TEST(DistTest, HsumFIntTest) {
        ALIGNED(16) int buf[4] = {1, 3, 5, 7};
        __m128i data = _mm_load_si128((__m128i *)buf);
        int h_sum = utils::HsumInt128(data);
        EXPECT_EQ(h_sum, 16);
    }


    void gen_rand_array(const std::size_t kDim, float *&a, float *&b) {
        static std::random_device r;
        static std::default_random_engine eng(r());
        static std::uniform_real_distribution<> frand(-1.0, 1.0);

        a = new float[kDim];
        b = new float[kDim];
        for (auto i=0; i<kDim; ++i) {
            a[i] = frand(eng);
            b[i] = frand(eng);
        }
    }

    TEST(DistTest, L2PSAVX512Test) {
        float *a;
        float *b;
        std::size_t dim = 3;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::L2SqrFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::L2SqrFloatAVX512(a, b, &dim) - utils::L2SqrNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 4;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::L2SqrFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::L2SqrFloatAVX512(a, b, &dim) - utils::L2SqrNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 7;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::L2SqrFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::L2SqrFloatAVX512(a, b, &dim) - utils::L2SqrNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 8;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::L2SqrFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::L2SqrFloatAVX512(a, b, &dim) - utils::L2SqrNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 32;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::L2SqrFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::L2SqrFloatAVX512(a, b, &dim) - utils::L2SqrNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        for (std::size_t d = 9; d<1025; ++d) {
            gen_rand_array(d, a, b);
            float avx_res = utils::L2SqrFloatAVX512(a, b, &d);
            float gt_res = utils::L2SqrNaive<float>(a, b, &d);
            std::cout << "Dim: " << d << ", dist: " << avx_res << std::endl;
            EXPECT_LE((avx_res - gt_res), 1e-3) << "avx512: " << avx_res << ", gt_res: " << gt_res;
            delete[] a;
            delete[] b;
        }
    }

    TEST(DistTest, IPPSAVX512Test) {
        float *a;
        float *b;
        std::size_t dim = 3;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::InnerProductFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::InnerProductFloatAVX512(a, b, &dim) - utils::IPNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 4;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::InnerProductFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::InnerProductFloatAVX512(a, b, &dim) - utils::IPNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;

        dim = 5;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::InnerProductFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::InnerProductFloatAVX512(a, b, &dim) - utils::IPNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;


        dim = 6;
        gen_rand_array(dim, a, b);
        float avx = utils::InnerProductFloatAVX512(a, b, &dim);
        float nv_ip = utils::IPNaive<float>(a, b, &dim);
        std::cout << "Dim: " << dim << ", dist: " << avx << ", nv_ip: " << nv_ip << std::endl;
        EXPECT_LE((avx - nv_ip), 1e-5);
        delete[] a;
        delete[] b;

        dim = 7;
        gen_rand_array(dim, a, b);
        std::cout << "Dim: " << dim << ", dist: " << utils::InnerProductFloatAVX512(a, b, &dim) << std::endl;
        EXPECT_LE((utils::InnerProductFloatAVX512(a, b, &dim) - utils::IPNaive<float>(a, b, &dim)), 1e-5);
        delete[] a;
        delete[] b;


        for (std::size_t d = 8; d<1025; ++d) {
            gen_rand_array(d, a, b);
            float avx_res = utils::InnerProductFloatAVX512(a, b, &d);
            float gt_res = utils::IPNaive<float>(a, b, &d);
            std::cout << "Dim: " << d << ", dist: " << avx_res << std::endl;
            EXPECT_LE((avx_res - gt_res), 1e-4) << "avx512: " << avx_res << ", gt_res: " << gt_res;
            delete[] a;
            delete[] b;
        }
    }

    TEST(DistTest, NormAVX512Test) {
        float *a;
        float *b;
        std::size_t dim = 3;
        gen_rand_array(dim, a, b);
        float avx_norm = utils::NormSqrtFloatAVX512(a, &dim);
        float nv_norm = utils::NormSqr<float>(a, &dim);
        EXPECT_LE((avx_norm - nv_norm), 1e-5);
        delete[] a;
        delete[] b;

        dim = 4;
        gen_rand_array(dim, a, b);
        avx_norm = utils::NormSqrtFloatAVX512(a, &dim);
        nv_norm = utils::NormSqr<float>(a, &dim);
        EXPECT_LE((avx_norm - nv_norm), 1e-5);
        delete[] a;
        delete[] b;




        dim = 5;
        gen_rand_array(dim, a, b);
        avx_norm = utils::NormSqrtFloatAVX512(a, &dim);
        nv_norm = utils::NormSqr<float>(a, &dim);
        EXPECT_LE((avx_norm - nv_norm), 1e-5);
        delete[] a;
        delete[] b;


        dim = 6;
        gen_rand_array(dim, a, b);
        avx_norm = utils::NormSqrtFloatAVX512(a, &dim);
        nv_norm = utils::NormSqr<float>(a, &dim);
        EXPECT_LE((avx_norm - nv_norm), 1e-5);
        delete[] a;
        delete[] b;

        dim = 7;
        gen_rand_array(dim, a, b);
        avx_norm = utils::NormSqrtFloatAVX512(a, &dim);
        nv_norm = utils::NormSqr<float>(a, &dim);
        EXPECT_LE((avx_norm - nv_norm), 1e-5);
        delete[] a;
        delete[] b;

        for (std::size_t d = 8; d<1025; ++d) {
            gen_rand_array(d, a, b);
            float avx_res = utils::NormSqrtFloatAVX512(a, &d);
            float gt_res = utils::NormSqr<float>(a, &d);
            std::cout << "Dim: " << d << ", a norm: " << avx_res << std::endl;
            EXPECT_LE((avx_res - gt_res), 1e-3) << "avx512: " << avx_res << ", gt_res: " << gt_res;
            avx_res = utils::NormSqrtFloatAVX512(b, &d);
            gt_res = utils::NormSqr<float>(b, &d);
             std::cout << "Dim: " << d << ", b norm: " << avx_res << std::endl;
            EXPECT_LE((avx_res - gt_res), 1e-3) << "avx512: " << avx_res << ", gt_res: " << gt_res;
            delete[] a;
            delete[] b;
        }
    }

    TEST(DistTest, L2SqrTest) {
        float *a;
        float *b;
        for (std::size_t d = 1; d<1026; ++d) {
            gen_rand_array(d, a, b);
            float gt = utils::L2SqrNaive<float>(a, b, &d);
            float avx512 = utils::L2SqrFloatAVX512(a, b, &d);
            std::cout << "Dim: " << d << ", l2sqr: " << gt << std::endl;
            EXPECT_LE(fabs(gt - avx512), 1e-3) << "avx512: " << avx512 << ", gt_res: " << gt;
            float avx = utils::L2SqrFloatAVX(a, b, &d);
            EXPECT_LE(fabs(avx - avx512), 1e-3) << "avx: " << avx << ", avx512: " << avx512;
            float sse = utils::L2SqrFloatSSE(a, b, &d);
            EXPECT_LE(fabs(avx - sse), 1e-3) << "avx: " << avx << ", sse: " << sse;
            delete[] a;
            delete[] b;
        }
    }

    TEST(DistTest, IpTest) {
        float *a;
        float *b;
        for (std::size_t d = 1; d<1026; ++d) {
            gen_rand_array(d, a, b);
            float gt = utils::IPNaive<float>(a, b, &d);
            float avx512 = utils::InnerProductFloatAVX512(a, b, &d);
            std::cout << "Dim: " << d << ", l2sqr: " << gt << std::endl;
            EXPECT_LE(fabs(gt - avx512), 1e-3) << "avx512: " << avx512 << ", gt_res: " << gt;
            float avx = utils::InnerProductFloatAVX(a, b, &d);
            EXPECT_LE(fabs(avx - avx512), 1e-3) << "avx: " << avx << ", avx512: " << avx512;
            float sse = utils::InnerProductFloatSSE(a, b, &d);
            EXPECT_LE(fabs(avx - sse), 1e-3) << "avx: " << avx << ", sse: " << sse;
            delete[] a;
            delete[] b;
        }
    }

    TEST(DistTest, NormTest) {
        float *a;
        float *b;
        for (std::size_t d = 1; d<1026; ++d) {
            gen_rand_array(d, a, b);
            float gt = utils::NormSqr<float>(a, &d);
            float avx512 = utils::NormSqrtFloatAVX512(a, &d);
            std::cout << "Dim: " << d << ", l2sqr: " << gt << std::endl;
            EXPECT_LE(fabs(gt - avx512), 1e-3) << "avx512: " << avx512 << ", gt_res: " << gt;
            float avx = utils::NormSqrtFloatAVX(a, &d);
            EXPECT_LE(fabs(avx - avx512), 1e-3) << "avx: " << avx << ", avx512: " << avx512;
            float sse = utils::NormSqrtFloatSSE(a, &d);
            EXPECT_LE(fabs(avx - sse), 1e-3) << "avx: " << avx << ", sse: " << sse;
            delete[] a;
            delete[] b;
        }
    }

}