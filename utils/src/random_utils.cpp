//
// Created by longxiang on 3/15/23.
//

#include "utils/random_utils.h"

namespace utils {
    unsigned random_uint(unsigned int min, unsigned int max) {
        // static std::mt19937 rng(seed);
        // return (rng() % max) + min;
        thread_local std::mt19937 mt(std::random_device{}());
        std::uniform_int_distribution<unsigned> dis(min, max);
        return dis(mt);
    }

        void GenRandomParallel(unsigned *addr, unsigned size, unsigned N) {
#pragma omp parallel for default(none) shared(addr, size, N)
        for (std::size_t i = 0; i < size; ++i) {
            addr[i] = utils::random_uint(0, N - size);
        }

        std::sort(addr, addr + N);

        for (std::size_t i = 1; i < size; ++i) {
            if (addr[i] <= addr[i-1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
    }

    void get_shuffle(const unsigned mul, const unsigned n, unsigned *res) {
        res = new unsigned[mul*n];
#pragma omp parallel for
        for (std::size_t i=0; i<mul; ++i) {
            for (std::size_t j=0; j<n; ++j) {
                res[i*n+j] = j;
            }
            thread_local std::minstd_rand mr(std::random_device{}());
            auto begin = res + i*n;
            auto end = begin+n;
            std::shuffle(begin, end, mr);
        }
    }


} // namespace utils
