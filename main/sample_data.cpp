#include "utils/io_utils.h"
#include "utils/dist_func.h"
#include "utils/sort_utils.h"
#include <random>
#include <algorithm>
#include <omp.h>

inline float* get_data(float *data, std::size_t idx) {
    return data + idx*100;
}

int main(int argc, char **argv) {
    const std::string kDataPath = argv[1];
    const size_t kSampleSize = atoi(argv[2]);
    const std::string kOutSampleData = argv[3];
    // const std::string kOutGtPath = argv[4];

    std::cout << "Sample Size: " << kSampleSize << std::endl;

    const std::size_t kTopk = 100;
    const std::size_t kDataDim = 100;

    float *data = nullptr;
    std::size_t data_num;
    FastRead(kDataPath, data, data_num);

    std::vector<std::size_t> sample_idx(data_num);
    std::iota(sample_idx.begin(), sample_idx.end(), 0);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::shuffle(sample_idx.begin(), sample_idx.end(), mt);

    float *sample_data = new float[kSampleSize * kDataDim];
#pragma omp parallel for default(none) shared(kSampleSize, data, sample_idx, kDataDim, sample_data)
    for (std::size_t idx=0; idx<kSampleSize; ++idx) {
        float *begin = data + sample_idx[idx] * kDataDim;
        float *end = begin + kDataDim;
        std::copy(begin, end, sample_data+ idx * kDataDim);
    }

//     unsigned *gt = new unsigned[kSampleSize * kDataDim];
// #pragma omp parallel for
//     for (std::size_t i = 0; i<kSampleSize; ++i) {
//         std::vector<unsigned> idx(kSampleSize);
//         std::vector<float> dist(kSampleSize);
//         std::iota(idx.begin(), idx.end(), 0);
//         for (std::size_t j = 0; j < kSampleSize; ++j) {
//             dist[j] = utils::L2SqrFloatAVX512(get_data(sample_data, i), get_data(sample_data, j), &kDataDim);
//         }
//         dist[i] = 10000;
//         utils::sort_2_array(dist.data(), idx.data(), kSampleSize);
//         std::copy(idx.begin(), idx.begin()+kTopk, gt+i*kTopk);
//     }

    WriteData<float>(kOutSampleData, sample_data, kSampleSize);
    // WriteData<unsigned >(kOutGtPath, gt, kSampleSize, kTopk);

    // delete[] gt;
    delete[] sample_data;
    delete[] data;
    return 0;
}
