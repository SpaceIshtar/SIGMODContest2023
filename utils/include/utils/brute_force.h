//
// Created by longxiang on 3/27/23.
//

#ifndef SIGMOD_CONTEST_BRUTE_FORCE_H
#define SIGMOD_CONTEST_BRUTE_FORCE_H

#include "utils/dist_func.h"
#include "utils/sort_utils.h"
#include <omp.h>
#include <numeric>

namespace utils {
    void brute_force(const float *data, const std::size_t num, const std::size_t dim, unsigned *res);

} // namespace utils

#endif //SIGMOD_CONTEST_BRUTE_FORCE_H
