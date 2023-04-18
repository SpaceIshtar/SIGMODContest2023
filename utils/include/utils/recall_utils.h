//
// Created by longxiang on 3/27/23.
//

#ifndef SIGMOD_CONTEST_RECALL_UTILS_H
#define SIGMOD_CONTEST_RECALL_UTILS_H

#include <cstddef>
#include <vector>

namespace utils {
    float get_recall(const unsigned *res, const unsigned res_row, const unsigned res_col, const unsigned *gt, const unsigned gt_col);
} // namespace utils

#endif //SIGMOD_CONTEST_RECALL_UTILS_H
