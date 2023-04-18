//
// Created by longxiang on 3/22/23.
//

#include "utils/random_utils.h"
#include "gtest/gtest.h"

namespace {
    TEST(RandUintTest, SingleThread) {
        for (auto i = 0; i<9; ++i) {
            std::cout << utils::random_uint(0, 100) << std::endl;
        }

        std::cout << "-------------" << std::endl << std::endl;

        for (auto i = 0; i<9; ++i) {
            std::cout << utils::random_uint(0, 100) << std::endl;
        }
    }

    TEST(RandUintTest, MultThread) {
        std::size_t size = 40;
        std::vector<uint32_t> a(size);

#pragma omp parallel for
        for (auto i = 0; i<size; ++i) {
            a[i] = utils::random_uint(0, 100);
        }

        for (auto i = 0; i<size; ++i) {
            std::cout << a[i] << std::endl;
        }
    }
}
