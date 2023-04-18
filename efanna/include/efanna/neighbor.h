//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <vector>
#include <mutex>
#include "hash_table7.h"
#include "util.h"
#include <unordered_set>
// #include <cuckoofilter/cuckoofilter_stable.h>
// #include <cuckoofilter/cuckoofilter.h>
// #include <cuckoofilter/simd-block-fixed-fpp.h>
// #include <cuckoofilter/hashing.h>
//#include "my_heap.h"

#define FAISS_PRAGMA_IMPRECISE_LOOP
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")

#define likely(x) __builtin_expect((bool) (x), true)
#define unlikely(x) __builtin_expect(bool(x), false)

namespace efanna2e {

    // std::vector<SimdBlockFilterFixed(1024)> filter;
     static std::minstd_rand rnng(1235); 

    struct DistId {
        unsigned id;
        float distance;

        DistId() = default;

        DistId(unsigned id, float dist): id{id}, distance{dist} {}

        inline bool operator<(const DistId &other) const {
            return distance < other.distance;
        }
    };

    struct Neighbor {
        unsigned id;
        float distance;
        bool flag;

        Neighbor() = default;

        Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

        inline bool operator<(const Neighbor &other) const {
            return distance < other.distance;
        }

        inline bool operator>(const Neighbor &other) const {
            return distance > other.distance;
        }

        const bool operator==(const Neighbor &other) const {
            return id == other.id;
        }

        struct HashFunction {
            size_t operator()(const Neighbor &neighbor) const {
                return std::hash<unsigned>()(neighbor.id);
            }
        };
    };

    typedef std::lock_guard<std::mutex> LockGuard;

    struct nhood {
        std::mutex lock;
        std::vector<Neighbor> pool;
//  heap<Neighbor> pool;
        unsigned M;
        unsigned n_new;
        unsigned n_old;

        // std::size_t insert_cnt = 0;
        // std::size_t for_cnt = 0;
        // std::size_t dist_cnt = 0;
        // std::size_t ex_in_cnt = 0;

        // SimdBlockFilterFixed<::hashing::TwoIndependentMultiplyShift> *filter;

        // cuckoofilter::CuckooFilter<unsigned, 8> *filter = nullptr;

        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;
        // std::vector<unsigned> rnn_old;
        // std::vector<unsigned> rnn_new;


        nhood() {
            M = 30;
            // nn_new.resize(60);
            // GenRandom(rnng, &nn_new[0], (unsigned) nn_new.size(), 10000000);
            // nn_new.reserve(60);
        }


        nhood(unsigned l, unsigned s) {
            M = s;
            pool.reserve(l);
            // filter = new SimdBlockFilterFixed<::hashing::TwoIndependentMultiplyShift>(8192);
            // filter = new cuckoofilter::CuckooFilter<unsigned , 8>(1000);
        }

        nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
            M = s;
            nn_new.resize(s * 2);
            GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
            nn_new.reserve(s * 2);
            pool.reserve(l);
            // filter = new SimdBlockFilterFixed<::hashing::TwoIndependentMultiplyShift>(8192);
            // filter = new cuckoofilter::CuckooFilter<unsigned, 8>(1000);
        }

        nhood(unsigned l, unsigned s, std::minstd_rand &rng, unsigned N) {
            M = s;
            nn_new.resize(s * 2);
            GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
            nn_new.reserve(s * 2);
            pool.reserve(l);
            // filter = new SimdBlockFilterFixed<::hashing::TwoIndependentMultiplyShift>(8192);
            // filter = new cuckoofilter::CuckooFilter<unsigned, 8>(1000);
        }

        nhood(const nhood &other) {
            M = other.M;
            std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
            // filter = other.filter;
        }

        nhood &operator=(const nhood &other) {
            if (this != &other) {
                M = other.M;
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
                // filter = other.filter;
            }

            return *this;
        }

        void insert(const DistId &dist_id) {
            for (auto &p: pool) {
                if (dist_id.id == p.id) return;
            }

            LockGuard guard(lock);
            if (pool.size() < pool.capacity()) {
                pool.emplace_back(dist_id.id, dist_id.distance, true);
                std::push_heap(pool.begin(), pool.end());
            } else {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(dist_id.id, dist_id.distance, true);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        void insert_b(unsigned id, float dist) {
            // if (dist > pool.front().distance) return;
            for (auto &i: pool) {
                if (id == i.id) return;
            }

            LockGuard guard(lock);
            if (pool.size() < pool.capacity()) {
//        pool.insert(Neighbor(id,dist,true));
                pool.emplace_back(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            } else {
//        pool.pop_insert(Neighbor(id,dist,true));
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        void insert(unsigned id, float dist) {
            if (dist > pool.front().distance) return;
            for (auto &i: pool) {
                if (id == i.id) return;
            }

            LockGuard guard(lock);
            if (pool.size() < pool.capacity()) {
//        pool.insert(Neighbor(id,dist,true));
                pool.emplace_back(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            } else {
//        pool.pop_insert(Neighbor(id,dist,true));
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        void insert_sorted(unsigned id, float dist){
            if (dist > pool.back().distance) return;
            unsigned i = 0;
            for (; i<pool.size();++i){
                if (dist <= pool[i].distance){
                    break;
                }
            }
            if (i == pool.size() || id == pool[i].id) return;
            LockGuard guard(lock);
            if (unlikely(pool.size()<pool.capacity())){
                pool.insert(pool.begin()+i,Neighbor(id,dist,true));
            }
            else{
                std::copy(pool.begin()+i,pool.end()-1,pool.begin()+i+1);
                pool[i] = Neighbor(id,dist,true);
            }
        }

        void insert_lock(unsigned id, float dist) {
//             LockGuard guard(lock);
//             ++insert_cnt;
//             if (dist > pool.front().distance) return;
//             ++dist_cnt;
//             for (auto &i: pool) {
//                 ++for_cnt;
//                 if (id == i.id) return;
//             }
//             ++ex_in_cnt;

//             if (pool.size() < pool.capacity()) {
// //        pool.insert(Neighbor(id,dist,true));
//                 pool.emplace_back(id, dist, true);
//                 std::push_heap(pool.begin(), pool.end());
//             } else {
// //        pool.pop_insert(Neighbor(id,dist,true));
//                 std::pop_heap(pool.begin(), pool.end());
//                 pool[pool.size() - 1] = Neighbor(id, dist, true);
//                 std::push_heap(pool.begin(), pool.end());
//             }
        }

        void insert_with_filter(unsigned id, float dist) {
//             if (dist > pool.front().distance) return;
//             if (filter->Find(id)) return;
//             for (auto &i: pool) {
//                 if (id == i.id) return;
//             }

//             // if (filter->Contain(id) == cuckoofilter::Ok) return;
//             LockGuard guard(lock);
//             if (pool.size() < pool.capacity()) {
//                 filter->Add(id);
// //      pool.insert(Neighbor(id,dist,true));
//                 pool.emplace_back(id, dist, true);
//                 std::push_heap(pool.begin(), pool.end());
//             } else {
//                 // filter->Delete(pool[0].id);
//                 filter->Add(id);
// //      pool.pop_insert(Neighbor(id,dist,true));
//                 std::pop_heap(pool.begin(), pool.end());
//                 pool[pool.size() - 1] = Neighbor(id, dist, true);
//                 std::push_heap(pool.begin(), pool.end());
//             }
        }

        FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN

        template<typename C>
        void join(C callback) const {
            unsigned nnNewSize = nn_new.size();
            if (likely(nnNewSize > 0)) {
                FAISS_PRAGMA_IMPRECISE_LOOP
                for (unsigned ii = 0; ii < nnNewSize - 1; ++ii) {
                    unsigned i = nn_new[ii];
                    for (unsigned jj = ii + 1; jj < nnNewSize; ++jj) {
                        unsigned j = nn_new[jj];
                        callback(i, j);
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
                unsigned i = nn_new[nnNewSize - 1];
                for (unsigned j: nn_old) {
                    callback(i, j);
                }
            }
        }

        FAISS_PRAGMA_IMPRECISE_FUNCTION_END

        template<typename C>
        void sp_join(C callback) const {
            for (unsigned const i: nn_new) {
                for (unsigned j: nn_old) {
                    callback(i, j);
                }
            }
        }
    };

    struct my_nhood {
        std::mutex lock;
        std::vector<Neighbor> pool;
        unsigned M;

        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;

        my_nhood() {}

        my_nhood(unsigned l, unsigned s) {
            M = s;
            pool.reserve(l);
        }

        my_nhood(const my_nhood &other) {
            std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
            pool.reserve(other.pool.capacity());
        }

        my_nhood &operator=(const my_nhood &other) {
            if (this != &other) {
                std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
                nn_new.reserve(other.nn_new.capacity());
                pool.reserve(other.pool.capacity());
            }

            return *this;
        }

        void insert(unsigned id, float dist) {
            LockGuard guard(lock);
            if (pool.size() == pool.capacity() && dist > pool.front().distance) return;
            for (unsigned i = 0; i < pool.size(); i++) {
                if (id == pool[i].id)return;
            }
            if (pool.size() < pool.capacity()) {
                pool.push_back(Neighbor(id, dist, true));
                std::push_heap(pool.begin(), pool.end());
            } else {
                std::pop_heap(pool.begin(), pool.end());
                pool[pool.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(pool.begin(), pool.end());
            }
        }

        template<typename C>
        void join(C callback) const {
            for (unsigned const i: nn_new) {
                for (unsigned const j: nn_new) {
                    if (i < j) {
                        callback(i, j);
                    }
                }
                for (unsigned j: nn_old) {
                    callback(i, j);
                }
            }
        }
    };

    struct LockNeighbor {
        std::mutex lock;
        std::vector<Neighbor> pool;
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1) {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)right = mid;
            else left = mid;
        }
        //check equal ID

        while (left > 0) {
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
        memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

}

#endif //EFANNA2E_GRAPH_H
