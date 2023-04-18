//
// Created by longxiang on 3/27/23.
//

#include "utils/recall_utils.h"

namespace utils {
    float get_recall(const unsigned *res, const unsigned res_row, const unsigned res_col, const unsigned *gt, const unsigned gt_col) {
        float acc = 0;
        for (std::size_t r=0; r<res_row; ++r) {
            std::vector<bool> flag(gt_col, true);
            for (std::size_t i = 0; i<gt_col; ++i) {
                for (std::size_t j = 0; j<gt_col; ++j) {
                    if(flag[j] && res[r*res_col+i] == gt[r*gt_col+j]) {
                        ++acc;
                        flag[j] = false;
                        break;
                    }
                }
            }
        }
        return acc / (res_row * gt_col);
    }
} // namespace utils
