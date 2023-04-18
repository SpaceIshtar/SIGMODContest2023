//
// Created by longxiang on 3/27/23.
//

#include "utils/brute_force.h"

namespace utils {
    void brute_force(const float *data, const std::size_t num, const std::size_t dim, unsigned *res) {

        float *val = new float[num * num];
        std::iota(res, res+num, 0);
        val[0] = 1000;

#pragma omp parallel for
        for (std::size_t i = 0; i < num - 1; ++i) {
            std::iota(res + (i+1) * num, res + (i+1) * num + num, 0);
            for (std::size_t j = i + 1; j < num; ++j) {
                float dist = utils::L2SqrFloatAVX512(&data[i * dim], &data[j * dim], &dim);
                val[i * num + j] = dist;
                val[j * num + i] = dist;
            }
            val[(i+1)*num + (i+1)] = 1000;
        }

#pragma omp parallel for
        for (std::size_t i = 0; i < num; ++i) {
            utils::sort_2_array(&val[i * num], &res[i * num], num);
        }

        delete[] val;
    }
}
