t //
// Created by longxiang on 3/20/23.
//

#include "utils/timer.h"
#include "utils/dist_func.h"
#include "utils/io_utils.h"
#include "utils/random_utils.h"
#include "utils/thread_safe_vec.h"
#include "hnsw/hnswalg.h"
#include <algorithm>
#include <thread>
#include <omp.h>

typedef std::vector<utils::ThreadSafeVector<uint32_t> > TSV;
typedef std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> HNSW_RES;

float *get_data(float *data, std::size_t idx) {
    return data + idx * 100;
}

void update_bucket(const std::size_t t_id, uint32_t t_iter, TSV &bucket, HNSW_RES &hnsw_res, const float kR,
                   const uint32_t data_num) {
    std::cout << "t_id: " << t_id << std::endl;
    std::cout << "t_iter: " << t_iter << std::endl;
    std::size_t begin_idx = t_id * t_iter;
    std::size_t end_idx = t_id * t_iter + t_iter;
    std::cout << "begin_idx: " << begin_idx << std::endl;
    std::cout << "end_idx: " << end_idx << std::endl;
    std::pair<float, hnswlib::labeltype> res1, res2;
    std::vector<std::size_t> update_buckets;
    update_buckets.reserve(t_iter);
    std::vector<bool> flag(data_num, false);
    for (std::size_t idx = begin_idx; idx < end_idx; ++idx) {
        res2 = hnsw_res[idx].top();

        hnsw_res[idx].pop();
        res1 = hnsw_res[idx].top();
        std::cout << "Idx: " << idx << std::endl;
        std::cout << "res2: " << res2.first << ", " << res2.second << std::endl;
        std::cout << "res1: " << res1.first << ", " << res1.second << std::endl;

//        bucket[res1.second].push_back((uint32_t) idx);
//        if (!flag[res1.second]) {
//            update_buckets.push_back(res1.second);
//            flag[res1.second] = true;
//        }
//        bucket[res2.second].push_back((uint32_t) idx);
//        if (!flag[res2.second]) {
//            update_buckets.push_back(res2.second);
//            flag[res2.second] = true;
//        }

//        if (res2.first < res1.first * kR) {
//            bucket[res2.second].push_back((uint32_t) idx);
//            if (!flag[res2.second]) {
//                update_buckets.push_back(res2.second);
//                flag[res2.second] = true;
//            }
//        }
    }
//    for (auto b: update_buckets) {
//        bucket[b].merge();
//    }
}

void merge_hnsw(std::vector<std::vector<uint32_t>> &graph, hnswlib::HierarchicalNSW<float> *hnsw) {

}

int main(int argc, char **argv) {
#ifdef RECALL_TEST
    omp_set_num_threads(32);
#endif
    const std::string kDataPath = argv[1];
    const std::string kResPath = argv[2];
    const uint32_t kSampleNum = atoi(argv[3]);
    const uint32_t kHnswM = atoi(argv[4]);
    const uint32_t kHnswEf = atoi(argv[5]);
    const uint32_t kHnswK = atoi(argv[6]);
    const float kR = atof(argv[7]);

    float *data = nullptr;
    uint32_t data_num, data_dim = 100;
    FastRead(kDataPath, data, data_num);

    std::vector<uint32_t> sample_list(kSampleNum);
    std::size_t idx;
#pragma omp parallel for default(none) shared(kSampleNum, data_num, sample_list) private(idx)
    for (idx = 0; idx < kSampleNum; ++idx) {
        sample_list[idx] = utils::random_uint(0, data_num);
    }

    std::sort(sample_list.begin(), sample_list.end());

    utils::Timer<std::chrono::milliseconds> timer;

    timer.reset();
    hnswlib::L2Space space(data_dim);
    hnswlib::HierarchicalNSW<float> *hnsw = new hnswlib::HierarchicalNSW<float>(&space, kSampleNum, kHnswM, kHnswEf);

    hnsw->addPoint(get_data(data, sample_list[0]), sample_list[0]);

#pragma omp parallel for default(none) shared(kSampleNum, data, sample_list, hnsw) private(idx)
    for (idx = 1; idx < kSampleNum; ++idx) {
        hnsw->addPoint(get_data(data, sample_list[idx]), sample_list[idx]);
    }
    std::cout << "HNSW Build Time: " << timer.getElapsedTime().count() << std::endl;

    std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> hnsw_res(data_num);

    timer.reset();
#pragma omp parallel for
    for (std::size_t n = 0; n < data_num; ++n) {
        hnsw_res[n] = hnsw->searchKnn(get_data(data, n), kHnswK);
    }

    std::cout << "HNSW Search Time: " << timer.getElapsedTime().count() << std::endl;

    uint32_t num_thread = 3;
    uint32_t num_iter = 6 / num_thread;
    std::vector<std::thread> threads;
    std::vector<utils::ThreadSafeVector<uint32_t> > bucket(kSampleNum);
    for (std::size_t t_id = 0; t_id < num_thread; ++t_id) {
        threads.emplace_back(std::thread([&bucket, &hnsw_res, t_id, num_iter, data_num, kR] {
            update_bucket(t_id, num_iter, bucket, hnsw_res, kR, data_num);
        }));
    }

    for (auto &thread: threads) {
        thread.join();
    }


//    int num_threads = 32;
//    int thread_itera = 10;
//    while (true) {
//        std::vector<std::thread> threads;
//        for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
//            threads.push_back(
//                    std::thread(
//                            [&] {
//                                for (auto iter = 0; iter < thread_itera; ++iter) {
//
//                                }
//                            }
//                    )
//            );
//        }
//        for (auto &thread:threads) {
//            thread.join();
//        }
//    }

}
