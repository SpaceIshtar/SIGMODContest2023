//
// Created by longxiang on 3/15/23.
//

#ifndef SIGMOD_CONTEST_RANDOM_UINT_H
#define SIGMOD_CONTEST_RANDOM_UINT_H

#include <random>
#include <algorithm>

namespace utils {
    unsigned random_uint(unsigned int min, unsigned int max);

    void GenRandomParallel(unsigned *addr, unsigned size, unsigned N);

    void GenRandArray() {

    }


} // namespace utils



#endif //SIGMOD_CONTEST_RANDOM_UINT_H
