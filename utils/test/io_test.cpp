#include "utils/io_utils.h"
#include "utils/timer.h"
#include "gtest/gtest.h"

namespace
{
    TEST(IoTest, LoadDataTest) {
        utils::Timer<std::chrono::microseconds> timer;
        char* file_path = "/home/longxiang/dataset/contest/contest-data-release-10m.bin";
        unsigned num, dim;
        float *data;
        timer.reset();
        load_data(file_path, data, num, dim);
        auto time = timer.getElapsedTime();
        std::cout << time.count() << std::endl;
    }

    TEST(IoTest, FastReadTest) {
        utils::Timer<std::chrono::microseconds> timer;
        std::string file_path = "/home/longxiang/dataset/contest/contest-data-release-10m.bin";
//        unsigned num, dim;
        float *data;
        timer.reset();
        FastRead(file_path, data);
        auto time = timer.getElapsedTime();
        std::cout << time.count() << std::endl;
    }

    TEST(IoTest, LoadReadTest) {
        char* file_path = "/home/longxiang/dataset/contest/contest-data-release-10m.bin";
        unsigned num, dim;
        float *l_data;
        load_data(file_path, l_data, num, dim);
        float *read_data;
        FastRead(file_path, read_data);
        utils::Timer<std::chrono::microseconds> timer;
        timer.reset();
        for (std::size_t i = 0; i < num; ++i) {
            for (std::size_t j = 0; j<dim; ++j) {
//                assert(l_data[i*dim+j] == read_data[i*dim+j]);
                if (l_data[i*dim+j] != read_data[i*dim+j]) {
                    std::cout << "i: " << i << ", j: " << j << std::endl;
                }
            }
        }
        std::cout << timer.getElapsedTime().count() << std::endl;
    }
} // namespace
