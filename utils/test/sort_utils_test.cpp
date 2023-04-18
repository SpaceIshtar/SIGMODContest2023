#include "utils/sort_utils.h"
#include "gtest/gtest.h"
#include <vector>

namespace {
    TEST(SortUtilsTest, Sort2ArrayTest) {
        std::vector<uint32_t> idx={0, 1, 2, 3, 4};
        std::vector<float> val{0.5, 0.1, 1.5, 0.8, 0.2};
        utils::sort_2_array(idx.data(), val.data(), 5);

        for (auto i =0; i<5; ++i) {
            std::cout << idx[i] << " ";
        }
        std::cout << std::endl;

    }
}