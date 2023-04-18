//
// Created by longxiang on 3/27/23.
//

#include "hnsw/hnswalg.h"
#include "utils/io_utils.h"
#include "utils/timer.h"
#include <algorithm>

int main(int argc, char **argv) {
    omp_set_num_threads(32);
    const std::string kDataPath = argv[1];
    const unsigned kHnswNum = atoi(argv[2]);
    const uint32_t kHnswM   = atoi(argv[3]);
    const uint32_t kHnswEf  = atoi(argv[4]);
    const uint32_t kRep     = atoi(argv[5]);
    const uint32_t kDataNum = atoi(argv[6]);

    float *data = nullptr;
    std::size_t data_num, data_dim = 100;
    FastRead(kDataPath, data, data_num);

    utils::Timer<std::chrono::milliseconds> timer;
    timer.reset();
    hnswlib::L2Space space(data_dim);
    hnswlib::HierarchicalNSW<float> *hnsw = new hnswlib::HierarchicalNSW<float>(&space, kHnswNum, kHnswM, kHnswEf);

    hnsw->addPoint(&data[0], 0);

#pragma omp parallel for
    for (std::size_t i=1; i<kHnswNum; ++i) {
        hnsw->addPoint(&data[i*data_dim], i);
    }
    std::cout << "HNSW Build Time: " << (float)timer.getElapsedTime().count() / 1000.0 << std::endl;

    std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> hnsw_res(kDataNum);
    timer.reset();
#pragma omp parallel for
    for (std::size_t i=0; i<kDataNum; ++i) {
        hnsw_res[i] = hnsw->searchKnn(&data[i*data_dim], kRep);
    }
    std::cout << "HNSW Search Time: " << (float)timer.getElapsedTime().count() / 1000.0 << std::endl;

    timer.reset();
    std::vector<std::vector<unsigned> > cluster(kHnswNum);
#pragma omp parallel for
    for (std::size_t i=0; i<kHnswNum; ++i) {
        cluster[i].reserve(int(data_num/kHnswNum)*kRep);
    }

    for (std::size_t i=0; i<kDataNum; ++i) {
        for (std::size_t j=0; j<kRep; ++j) {
            auto res = hnsw_res[i].top();
            cluster[res.second].emplace_back(i);
            hnsw_res[i].pop();
        }
    }
    std::cout << "Cluster Time: " << (float)timer.getElapsedTime().count() / 1000.0 << std::endl;

    unsigned max_cluster = 0;
    unsigned min_cluster = data_num;
    for (std::size_t i=0; i<kHnswNum; ++i) {
        if (cluster[i].size() < min_cluster) {
            min_cluster = cluster[i].size();
        } else if (cluster[i].size() > max_cluster) {
            max_cluster = cluster[i].size();
        }
    }
    std::cout << "Max Cluster: " << max_cluster << std::endl;
    std::cout << "Min Cluster: " << min_cluster << std::endl;

    std::vector<unsigned> cluster_size(kHnswNum);
//    float avg_size = 0;
    for (std::size_t i=0; i<kHnswNum; ++i) {
        cluster_size[i] = cluster[i].size();
//        avg_size += cluster_size[i];
    }
    std::sort(cluster_size.begin(), cluster_size.end());
//    std::cout << "Avg Cluster Size: " << avg_size / kHnswNum << std::endl;
//    for (std::size_t i=0; i<kHnswNum; ++i) {
//        std::cout << cluster_size[i] << "     ";
//    }
//    std::cout << std::endl;

    unsigned k1 = 0, k10 = 0;
    for (std::size_t i=0; i<kHnswNum; ++i) {
        if (cluster_size[i] < 1000) {
            ++k1;
        } else if (cluster_size[i] > 10000) {
            ++k10;
        }
    }
    std::cout << "K1: " << k1 << std::endl;
    std::cout << "K10: " << k10 << std::endl;

    return 0;
}