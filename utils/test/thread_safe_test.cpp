//
// Created by longxiang on 3/21/23.
//
#include "utils/thread_safe_vec.h"
#include "gtest/gtest.h"
#include <thread>

namespace {
    void push_back(utils::ThreadSafeVector<uint32_t> &vec, uint32_t t_id, uint32_t t_iter) {
        uint32_t begin = t_id * t_iter;
        uint32_t end = t_id * t_iter + t_iter;
        std::cout << "t_id: " << t_id << std::endl;
        std::cout << "t_iter: " << t_iter << std::endl;
        for (auto i=begin; i<end; ++i) {
            std::cout << "i: " << i << std::endl;
            vec.push_back(i);
        }
        vec.merge();
    }

//    void push_back(std::vector<utils::ThreadSafeVector<uint32_t>> &vec, uint32_t t_id, uint32_t t_iter) {
//        uint32_t begin = t_id * t_iter;
//        uint32_t end = t_id * t_iter + t_iter;
//        std::cout << "t_id: " << t_id << std::endl;
//        std::cout << "t_iter: " << t_iter << std::endl;
//        for (auto i=begin; i<end; ++i) {
//            std::cout << "i: " << i << std::endl;
//            vec.push_back(i);
//        }
//        vec.merge();
//    }

    TEST(ThreadSafeVecTest, SingleTest) {
        utils::ThreadSafeVector<uint32_t> vec;

        uint32_t  num_threads = 10;
        uint32_t  thread_itera = 5;
        std::vector<std::thread> threads;
        for (uint32_t t_id = 0; t_id < num_threads; ++t_id) {
            threads.emplace_back(std::thread([&vec,t_id,thread_itera]{
                push_back(vec,t_id, thread_itera);
            }));
        }
        for (auto  &thread:threads) {
            thread.join();
        }
        std::vector<uint32_t> res;
        vec.get_vector(res);
        for (auto r:res) {
            std::cout << r <<" " << std::endl;
        }
    }

    TEST(ThreadSafeVecTest, MultTest) {
        std::vector<utils::ThreadSafeVector<uint32_t>> vecs(10);

    }

}