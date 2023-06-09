//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#include <efanna/index_graph.h>
#include <efanna/exceptions.h>
#include <efanna/parameters.h>
#include <omp.h>
#include <set>
#include <unordered_set>
#include <queue>

namespace efanna2e {
#define _CONTROL_NUM 100

    IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer)
            : Index(dimension, n, m),
              initializer_{initializer} {
        init_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        assert(dimension == initializer->GetDimension());
    }

    IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, const Parameters &param)
            : Index(dimension, n, m) {
        init_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        unsigned R = param.Get<unsigned>("R");
        unsigned L = param.Get<unsigned>("L");
        pool_len_ = (R + L) * 2;
        std::cout << "pool_len_: " << pool_len_ << std::endl;
        thread_pool_size_ = pool_len_ * pool_len_;
        std::size_t thread_num = 32;
        std::cout << "Thread Number: " << thread_num << std::endl;
        dist_pool_.resize(thread_num);
#pragma omp parallel for
        for (std::size_t t_id=0; t_id<thread_num; ++t_id) {
            dist_pool_[t_id].resize(thread_pool_size_ );
        }
    }


    IndexGraph::~IndexGraph() {}

    void IndexGraph::join_sorted(){
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (auto &nnhd:graph_) {
            nnhd.join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = distfunc(data_ + i * dimension_, data_ + j * dimension_, &dimension_);
                    graph_[i].insert_sorted(j, dist);
                    graph_[j].insert_sorted(i, dist);
                }
            });
        }
    }

    void IndexGraph::update_sorted(unsigned L,unsigned S, unsigned R, unsigned iteration){
#pragma omp parallel for
//   for (unsigned n = 0; n < nd_; ++n) {
  for (auto &nn : graph_) {
    // auto &nn = graph_[n];
    std::vector<unsigned>().swap(nn.nn_new);
    std::vector<unsigned>().swap(nn.nn_old);
    // std::sort(nn.pool.begin(), nn.pool.end());
    if (nn.pool.capacity() < L) nn.pool.reserve(L);
    unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
    // unsigned c = 0;
    nn.n_new = 0;
    unsigned l = 0;
    while ((l < maxl) && (nn.n_new < S + 1)) {
      if (nn.pool[l].flag) ++nn.n_new;
      ++l;
    }
    nn.M = l;
    nn.n_old = nn.M - nn.n_new;
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      if (nn.flag) {
        nn_new.push_back(nn.id);
        // nn.flag = false;
      } else {
        nn_old.push_back(nn.id);
      }
    }
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      auto &nhood_o = graph_[nn.id];  // nn on the other side of the edge
      if (nn.flag) {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_new.size() < nhood_o.n_new + R)nhood_o.nn_new.push_back(n);
        }
        nn.flag = false;
      } else {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_old.size() < nhood_o.n_old + R)nhood_o.nn_old.push_back(n);
        }
      }
    }
  }
    }

    void IndexGraph::update_guodu(unsigned L, unsigned S, unsigned R, unsigned iteration){
#pragma omp parallel for
//   for (unsigned n = 0; n < nd_; ++n) {
  for (auto &nn : graph_) {
    // auto &nn = graph_[n];
    std::vector<unsigned>().swap(nn.nn_new);
    std::vector<unsigned>().swap(nn.nn_old);
    std::sort(nn.pool.begin(), nn.pool.end());
    if (nn.pool.capacity() < L) nn.pool.reserve(L);
    unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
    // unsigned c = 0;
    nn.n_new = 0;
    unsigned l = 0;
    while ((l < maxl) && (nn.n_new < S + 1)) {
      if (nn.pool[l].flag) ++nn.n_new;
      ++l;
    }
    nn.M = l;
    nn.n_old = nn.M - nn.n_new;
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      if (nn.flag) {
        nn_new.push_back(nn.id);
        // nn.flag = false;
      } else {
        nn_old.push_back(nn.id);
      }
    }
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      auto &nhood_o = graph_[nn.id];  // nn on the other side of the edge
      if (nn.flag) {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_new.size() < nhood_o.n_new + R)nhood_o.nn_new.push_back(n);
        }
        nn.flag = false;
      } else {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_old.size() < nhood_o.n_old + R)nhood_o.nn_old.push_back(n);
        }
      }
    }
  }
    }

    void IndexGraph::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (auto &nnhd:graph_) {
             nnhd.join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = distfunc(data_ + i * dimension_, data_ + j * dimension_, &dimension_);
                    graph_[i].insert(j, dist);
                    graph_[j].insert(i, dist);
                }
            });
        }
    }


FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
    void IndexGraph::join_batch() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (std::size_t n = 0; n < nd_; ++n) {
            auto &n_new = graph_[n].nn_new;
            std::size_t n_new_size = n_new.size();

            if (n_new_size >= pool_len_) {
                std::cout << "n_new_size >= pool_len" << std::endl;
                assert(false);
            }

            if (unlikely(n_new_size == 0)) continue;
            int thread_id = omp_get_thread_num();
            auto &d_pool = dist_pool_[thread_id];

           FAISS_PRAGMA_IMPRECISE_LOOP
            for (std::size_t i = 0; i < n_new_size - 1; ++i) {
                unsigned i_id = n_new[i];
                d_pool[i*n_new_size+i].distance = 1000;
                d_pool[i*n_new_size+i].id = i_id;
                for (std::size_t j = i + 1; j < n_new_size; ++j) {
                    unsigned j_id = n_new[j];
                    if (likely(i_id != j_id)) {
                        float dist = distfunc(data_ + i_id * dimension_, data_ + j_id * dimension_, &dimension_);
                        d_pool[i * n_new_size + j].distance = dist;
                        d_pool[i * n_new_size + j].id = j_id;
                        d_pool[j * n_new_size + i].distance = dist;
                        d_pool[j * n_new_size + i].id = i_id;
                   } else {
                       d_pool[i*n_new_size+j].distance = 1000;
                       d_pool[i*n_new_size+j].id = j_id;
                       d_pool[j*n_new_size+i].distance = 1000;
                       d_pool[j*n_new_size+i].id = i_id;
                   }
                }
            }
            d_pool[n_new_size*n_new_size-1].distance = 1000;
            d_pool[n_new_size*n_new_size-1].id = n_new[n_new_size-1];

           FAISS_PRAGMA_IMPRECISE_LOOP
            for (std::size_t i=0; i<n_new_size; ++i) {
                unsigned i_id = n_new[i];
                auto &g = graph_[i_id];
                for (std::size_t j=0; j<n_new_size; ++j) {
                    if (g.pool.front().distance < d_pool[i*n_new_size+j].distance) continue;
                    g.insert(d_pool[i*n_new_size+j]);
                }
            }
            // std::cout << "n_new insert succ" << std::endl;

            auto &n_old = graph_[n].nn_old;
            std::size_t n_old_size = n_old.size();

           FAISS_PRAGMA_IMPRECISE_LOOP
            for (std::size_t i=0; i<n_new_size; ++i) {
                unsigned i_id = n_new[i];
                for (std::size_t j=0; j<n_old_size; ++j) {
                    unsigned j_id = n_old[j];
                    if (likely(i_id != j_id)) {
                        float dist = distfunc(data_ + i_id * dimension_, data_ + j_id * dimension_, &dimension_);
                        d_pool[i * n_old_size + j].distance = dist;
                        d_pool[i * n_old_size + j].id = j_id;
                   } else {
                       d_pool[i*n_old_size+j].distance = 1000;
                       d_pool[i*n_old_size+j].id = j_id;
                   }
                }
            }

           FAISS_PRAGMA_IMPRECISE_LOOP
            for (std::size_t i=0; i<n_new_size; ++i) {
                unsigned i_id = n_new[i];
                auto &g = graph_[i_id];
                for (std::size_t j=0; j<n_old_size; ++j) {
                    if (g.pool.front().distance < d_pool[i*n_old_size+j].distance) continue;
                    g.insert(d_pool[i*n_old_size+j]);

                    auto &g_j = graph_[n_old[j]];
                    if (g_j.pool.front().distance<d_pool[i*n_old_size+j].distance) continue;
                    g_j.insert_b(i_id, d_pool[i*n_old_size+j].distance);
                }
            }
            // std::cout << "n_old insert succ" << std::endl;
        }
    }
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

void IndexGraph::update(unsigned L, unsigned S, unsigned R, unsigned iteration) {
#pragma omp parallel for
//   for (unsigned n = 0; n < nd_; ++n) {
  for (auto &nn : graph_) {
    // auto &nn = graph_[n];
    std::vector<unsigned>().swap(nn.nn_new);
    std::vector<unsigned>().swap(nn.nn_old);
    std::sort(nn.pool.begin(), nn.pool.end());
    if (nn.pool.capacity() < L) nn.pool.reserve(L);
    unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
    // unsigned c = 0;
    nn.n_new = 0;
    unsigned l = 0;
    while ((l < maxl) && (nn.n_new < S + 1)) {
      if (nn.pool[l].flag) ++nn.n_new;
      ++l;
    }
    nn.M = l;
    nn.n_old = nn.M - nn.n_new;
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      if (nn.flag) {
        nn_new.push_back(nn.id);
        // nn.flag = false;
      } else {
        nn_old.push_back(nn.id);
      }
    }
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nnhd = graph_[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (unsigned l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      auto &nhood_o = graph_[nn.id];  // nn on the other side of the edge
      if (nn.flag) {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_new.size() < nhood_o.n_new + R)nhood_o.nn_new.push_back(n);
        }
        nn.flag = false;
      } else {
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if(nhood_o.nn_old.size() < nhood_o.n_old + R)nhood_o.nn_old.push_back(n);
        }
      }
    }
    std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
  }
}

    void IndexGraph::NNDescent_batch(const Parameters &parameters) {
        unsigned iter = parameters.Get<unsigned>("iter");
        unsigned S = parameters.Get<unsigned>("S");
        unsigned L = parameters.Get<unsigned>("L");
        unsigned R = parameters.Get<unsigned>("R");
        for (unsigned it = 0; it < iter; it++) {
            std::cout << "iter: " << it << std::endl;
            auto it_start = std::chrono::high_resolution_clock::now();
//            timer.tuck("Iteration start");
            join_batch();
            auto join_end = std::chrono::high_resolution_clock::now();
//            timer.tuck("Join ends");
            update(L,S,R, it);
            auto update_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> join_diff = join_end - it_start;
            std::chrono::duration<double> update_diff = update_end - join_end;
            timer.tuck("Iteration " + std::to_string(it) + " Ends");
            std::cout << "In Iteration " << it << ", Join cost: " << join_diff.count() << ", Update cost: "
                      << update_diff.count() << std::endl;
           eval_recall();
        }
    }

    void IndexGraph::Build_batch(size_t n, const float *data, const Parameters &parameters) {
    }


    void IndexGraph::NNDescent(const Parameters &parameters) {
        unsigned iter = parameters.Get<unsigned>("iter");
        unsigned K = parameters.Get<unsigned>("K");
        for (unsigned it = 0; it < 5; it++) {
            std::cout << "iter: " << it << std::endl;
            auto it_start = std::chrono::high_resolution_clock::now();
            join();
            auto join_end = std::chrono::high_resolution_clock::now();
            update(120,30,200,it);
            auto update_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> join_diff = join_end - it_start;
            std::chrono::duration<double> update_diff = update_end - join_end;
            timer.tuck("Iteration " + std::to_string(it) + " Ends");
            std::cout << "In Iteration " << it << ", Join cost: " << join_diff.count() << ", Update cost: "
                      << update_diff.count() << std::endl;
           eval_recall();
        }

         std::cout << "iter: " << 5 << std::endl;
         auto it_start = std::chrono::high_resolution_clock::now();
         join();
         auto join_end = std::chrono::high_resolution_clock::now();
         update(240,30,400, 5);
         auto update_end = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> join_diff = join_end - it_start;
         std::chrono::duration<double> update_diff = update_end - join_end;
         timer.tuck("Iteration 5 Ends");
         std::cout << "In Iteration " << 5 << ", Join cost: " << join_diff.count() << ", Update cost: "
                   << update_diff.count() << std::endl;
        eval_recall();

        std::cout << "iter: " << 6 << std::endl;
        it_start = std::chrono::high_resolution_clock::now();
        join();
        join_end = std::chrono::high_resolution_clock::now();
        update(240,30,400, 6);
        update_end = std::chrono::high_resolution_clock::now();
        join_diff = join_end - it_start;
        update_diff = update_end - join_end;
        timer.tuck("Iteration 6 Ends");
        std::cout << "In Iteration " << 6 << ", Join cost: " << join_diff.count() << ", Update cost: "
                  << update_diff.count() << std::endl;
        eval_recall();

        std::cout << "iter: " << 7 << std::endl;
        it_start = std::chrono::high_resolution_clock::now();
        join();
        join_end = std::chrono::high_resolution_clock::now();
        update(240,30,400, 7);
        update_end = std::chrono::high_resolution_clock::now();
        join_diff = join_end - it_start;
        update_diff = update_end - join_end;
        timer.tuck("Iteration 7 Ends");
        std::cout << "In Iteration " << 7 << ", Join cost: " << join_diff.count() << ", Update cost: "
                  << update_diff.count() << std::endl;
        eval_recall();


        std::cout << "iter: " << 8 << std::endl;
        it_start = std::chrono::high_resolution_clock::now();
        join();
        join_end = std::chrono::high_resolution_clock::now();
        update_guodu(240,30,400, 8);
        update_end = std::chrono::high_resolution_clock::now();
        join_diff = join_end - it_start;
        update_diff = update_end - join_end;
        timer.tuck("Iteration 8 Ends");
        std::cout << "In Iteration " << 8 << ", Join cost: " << join_diff.count() << ", Update cost: "
                  << update_diff.count() << std::endl;
        eval_recall();

        for (unsigned it = 9; it<iter;++it){
            std::cout << "iter: " << it << std::endl;
            auto it_start = std::chrono::high_resolution_clock::now();
            join_sorted();
            auto join_end = std::chrono::high_resolution_clock::now();
            update_sorted(240,30,400,it);
            auto update_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> join_diff = join_end - it_start;
            std::chrono::duration<double> update_diff = update_end - join_end;
            timer.tuck("Iteration " + std::to_string(it) + " Ends");
            std::cout << "In Iteration " << it << ", Join cost: " << join_diff.count() << ", Update cost: "
                      << update_diff.count() << std::endl;
           eval_recall();
        }
        
    }

    void IndexGraph::generate_control_set(std::vector<unsigned> &c,
                                          std::vector<std::vector<unsigned> > &v,
                                          unsigned N) {
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                float dist = distance_->compare(data_ + c[i] * dimension_, data_ + j * dimension_, dimension_);
                tmp.push_back(Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    void IndexGraph::eval_recall() {
//         unsigned K = 100, sample_num = 100000;
//         float mean_acc = 0;
//         for (unsigned i = 0; i < sample_num; i++) {
//             unsigned data_id = i;
//             float acc = 0;
// //            auto &g1 = graph_[i].pool;
//             auto &g1 = graph_[data_id].pool;
//             auto g = std::vector<efanna2e::Neighbor>(g1);
//             unsigned k = K < g.size() ? K : g.size();
//             std::sort(g.begin(), g.end());
//             emhash8::HashSet<unsigned> gSet;
//             for (auto it = g.begin(); it != g.end(); ++it) {
//                 gSet.emplace((*it).id);
//                 if (gSet.size() == k) break;
//             }

//             for (unsigned h = 0; h < K; h++) {
//                 if (gSet.contains(gt_[i * K + h])) {
// //                if (g[j].id == gt_[i * K + h]) {
//                     acc++;
//                 }
//             }

//             mean_acc += acc / K;
//         }
//         std::cout << "recall : " << mean_acc / (float) (sample_num) << std::endl;
    }

    void IndexGraph::eval_final_recall() {
//         unsigned K = 100, sample_num = 100000;
//         float mean_acc = 0;
//         for (unsigned i = 0; i < sample_num; i++) {
//             unsigned data_id = i;
//             float acc = 0;
// //            auto &g1 = graph_[i].pool;
//             auto &g1 = final_graph_[data_id];
//             auto g = std::vector<unsigned>(g1);
//             unsigned k = K < g.size() ? K : g.size();
//             std::partial_sort(g.begin(), g.begin() + k, g.end());
//             for (unsigned j = 0; j < k; j++) {
//                 for (unsigned h = 0; h < K; h++) {
//                     if (g[j] == gt_[i * K + h]) {
//                         acc++;
//                         break;
//                     }
//                 }
//             }
//             mean_acc += acc / K;
//         }
//         for (unsigned i = 0; i < sample_num; i++) {
//             unsigned data_id = nd_ - (sample_num - i);
//             float acc = 0;
// //            auto &g1 = graph_[i].pool;
//             auto &g1 = final_graph_[data_id];
//             auto g = std::vector<unsigned>(g1);
//             unsigned k = K < g.size() ? K : g.size();
//             std::partial_sort(g.begin(), g.begin() + k, g.end());
//             for (unsigned j = 0; j < k; j++) {
//                 for (unsigned h = 0; h < K; h++) {
//                     if (g[j] == gt2_[i * K + h]) {
//                         acc++;
//                         break;
//                     }
//                 }
//             }
//             mean_acc += acc / K;
//         }
//         std::cout << "recall : " << mean_acc / (2 * sample_num) << std::endl;
    }

    void IndexGraph::InitializeGraph(const Parameters &parameters) {
        std::cout << "Begin Init" << std::endl;

        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");

        timer.tick();
        std::minstd_rand rng(init_seed);
        graph_.resize(nd_);

#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            graph_[i].nn_new.resize(60);
            GenRandom(rng, &graph_[i].nn_new[0], 60, nd_);
        }
        timer.tuck("graph init");

#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            std::vector<unsigned> tmp(S + 1);
            GenRandom(rng, tmp.data(), S + 1, nd_);
            graph_[i].pool.reserve(L);
            for (unsigned j = 0; j < S; j++) {
                unsigned id = tmp[j];
                if (id == i) continue;
                float dist = distfunc(data_ + i * dimension_, data_ + id * dimension_, &dimension_);
                graph_[i].pool.emplace_back(id, dist, true);
            }
            std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
        }
    }

    void IndexGraph::InitializeGraph_Refine(const Parameters &parameters) {
    }

    void IndexGraph::RefineBySearch(const Parameters &parameters) {
    }

    void IndexGraph::RefineByBFS(const efanna2e::Parameters &parameters) {
    }

    void IndexGraph::RefineByHubness(const efanna2e::Parameters &parameters) {
//         unsigned cluster_num = 10000;
//         unsigned K = parameters.Get<unsigned>("K");
//         unsigned L = parameters.Get<unsigned>("L");
//         unsigned ef = 3 * K;
//         unsigned center_num = 100;
//         std::vector<std::pair<unsigned, unsigned>> hubness(nd_);
// #pragma omp parallel for
//         for (unsigned i = 0; i < nd_; ++i) {
//             hubness[i] = std::pair<unsigned, unsigned>(0, i);
//         }
//         for (unsigned i = 0; i < nd_; ++i) {
// #pragma omp parallel for
//             for (auto &neighbor: graph_[i].pool) {
//                 hubness[neighbor.id].first += 1;
//             }
//         }
//         timer.tuck("Hubness Count Ends");
//         std::sort(hubness.begin(), hubness.end());
//         emhash8::HashSet<unsigned> hubness_set;
//         float *hubness_data = new float[cluster_num * dimension_];
//         std::vector<unsigned> hub_id2ori_id(cluster_num);
// #pragma omp parallel for
//         for (unsigned i = 0; i < cluster_num; ++i) {
//             unsigned id = hubness[nd_ - i].second;
//             memcpy(hubness_data + i * dimension_, data_ + id * dimension_, dimension_ * sizeof(float));
//             hub_id2ori_id[i] = id;
//         }
//         for (auto id: hub_id2ori_id) {
//             hubness_set.emplace(id);
//         }
//         timer.tuck("Hubness Data Copy Ends");
//         std::vector<emhash8::HashSet<unsigned>> node2hubness(nd_, emhash8::HashSet<unsigned>());
//         std::vector<std::pair<unsigned, unsigned>> depth3_count(nd_);
// #pragma omp parallel for
//         for (unsigned i = 0; i < nd_; i++) {
//             emhash8::HashSet<unsigned> visit_set; // (id)
//             unsigned count = 0;
//             std::queue<std::pair<unsigned, unsigned>> queue; // (id, depth)
//             for (auto &neighbor: graph_[i].pool) {
//                 visit_set.emplace(neighbor.id);
//                 queue.emplace(neighbor.id, 1);
//             }
//             while (!queue.empty()) {
//                 auto pair = queue.front();
//                 queue.pop();
//                 if (hubness_set.contains(pair.first)) {
//                     node2hubness[i].emplace(pair.first);
//                 }
//                 for (auto &neighbor: graph_[pair.first].pool) {
//                     if (!visit_set.contains(neighbor.id)) {
//                         visit_set.emplace(neighbor.id);
//                         if (pair.second < 3) {
//                             queue.emplace(neighbor.id, pair.second + 1);
//                         } else {
//                             ++count;
//                         }
//                     }
//                 }
//                 depth3_count[i] = std::pair<unsigned, unsigned>(count, i);
//             }
//         }
//         for (unsigned i = 0; i < 1000; i++) {
//             std::cout << "Node " << depth3_count[i].second << " has " << depth3_count[i].first << " depth-3 neighbors."
//                       << std::endl;
//         }
//         timer.tuck("Find hubnesses Ends.");

//         exit(0);
    }

    void IndexGraph::Build(size_t n, const float *data, const Parameters &parameters) {
        data_ = data;
        assert(initializer_->HasBuilt());

//        timer.tick();
        InitializeGraph(parameters);
        timer.tuck("Initialize");

        NNDescent(parameters);
        unsigned K = parameters.Get<unsigned>("K");
        final_graph_ = new unsigned[nd_*K];
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            for (unsigned j = 0; j < K; j++) {
                final_graph_[i*K+j] = graph_[i].pool[j].id;
            }
            std::vector<Neighbor>().swap(graph_[i].pool);
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
        }
        std::vector<nhood>().swap(graph_);
        has_built = true;
    }

    void IndexGraph::Search(
            const float *query,
            const float *x,
            size_t K,
            const Parameters &parameter,
            unsigned *indices) {
    }

    void IndexGraph::mergePool(IndexGraph *index) {
        assert(nd_ == index->nd_);
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; ++i) {
            for (auto &neighbor: index->graph_[i].pool) {
                if (neighbor.distance < graph_[i].pool.front().distance) {
                    graph_[i].insert(neighbor.id, neighbor.distance);
                }
            }
        }
    }

    void IndexGraph::Save(const char *filename) {
        FILE *fp = nullptr;    
        fp = fopen(filename, "wb");    
        fwrite(final_graph_, sizeof(unsigned), nd_*100, fp);    
        fclose(fp);
    }

    void IndexGraph::Load(const char *filename) {
    }

    void IndexGraph::parallel_graph_insert(unsigned id, Neighbor nn, LockGraph &g, size_t K) {
        LockGuard guard(g[id].lock);
        size_t l = g[id].pool.size();
        if (l == 0)g[id].pool.push_back(nn);
        else {
            g[id].pool.resize(l + 1);
            g[id].pool.reserve(l + 1);
            InsertIntoPool(g[id].pool.data(), (unsigned) l, nn);
            if (g[id].pool.size() > K)g[id].pool.reserve(K);
        }

    }

    void IndexGraph::get_neighbor_to_add(const float *point,
                                         const Parameters &parameters,
                                         LockGraph &g,
                                         std::mt19937 &rng,
                                         std::vector<Neighbor> &retset,
                                         unsigned n_new) {
        const unsigned L = parameters.Get<unsigned>("L_ADD");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        GenRandom(rng, init_ids.data(), L / 2, n_new);
        for (unsigned i = 0; i < L / 2; i++)init_ids[i] += nd_;

        GenRandom(rng, init_ids.data() + L / 2, L - L / 2, (unsigned) nd_);

        unsigned n_total = (unsigned) nd_ + n_new;
        std::vector<char> flags(n_new + n_total);
        memset(flags.data(), 0, n_total * sizeof(char));
        for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float dist = distance_->compare(data_ + dimension_ * id, point, (unsigned) dimension_);
            retset[i] = Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin() + L);
        int k = 0;
        while (k < (int) L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;

                LockGuard guard(g[n].lock);//lock start
                for (unsigned m = 0; m < g[n].pool.size(); ++m) {
                    unsigned id = g[n].pool[m].id;
                    if (flags[id])continue;
                    flags[id] = 1;
                    float dist = distance_->compare(point, data_ + dimension_ * id, (unsigned) dimension_);
                    if (dist >= retset[L - 1].distance)continue;
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);

                    //if(L+1 < retset.size()) ++L;
                    if (r < nk)nk = r;
                }
                //lock to here
            }
            if (nk <= k)k = nk;
            else ++k;
        }


    }

    void IndexGraph::compact_to_Lockgraph(LockGraph &g) {

    }

    void IndexGraph::BruteForce(const Parameters &parameters) {
        const unsigned W = parameters.Get<unsigned>("W");
        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned clusters_num = cluster_ids.size();

        for (unsigned cls = 0; cls < clusters_num; cls++) {
            Neighbor *neighbors = new Neighbor[W * W];
            auto indices = cluster_ids[cls];
            const unsigned w = indices.size();
#pragma omp parallel for
            for (unsigned i = 0; i < w - 1; i++) {
                for (unsigned j = i + 1; j < w; j++) {
                    float dist = distfunc(data_ + indices[i] * dimension_, data_ + indices[j] * dimension_,
                                          &dimension_);
                    neighbors[i * W + j] = Neighbor(indices[j], dist, true);
                    neighbors[j * W + i] = Neighbor(indices[i], dist, true);
                }
            }

            const unsigned K = L < w ? L : w;
#pragma omp parallel for
            for (unsigned i = 0; i < w; i++) {
                // std::partial_sort(neighbors + i*W, neighbors + i*W + K + 1, neighbors + i*W + w);
                for (unsigned j = 0; j < w; j++) {
                    // graph_[indices[i]].pool.push_back(neighbors[i*W+j]);
                    auto &pool = graph_[indices[i]].pool;
                    if (std::find(pool.begin(), pool.end(), neighbors[i * W + j]) == pool.end() && i != j) {
                        if (pool.size() < L) {
                            pool.push_back(neighbors[i * W + j]);
                            std::push_heap(pool.begin(), pool.end());
                        } else {
                            std::pop_heap(pool.begin(), pool.end());
                            pool[pool.size() - 1] = neighbors[i * W + j];
                            std::push_heap(pool.begin(), pool.end());
                        }
                    }
                }
            }

            delete[] neighbors;
        }
    }

    void IndexGraph::RandomMultipleDivision(const Parameters &parameters) {
        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");

        // initialization
        graph_.resize(nd_);
        std::mt19937 rng(rand());
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            graph_[i] = nhood(L, S, rng, (unsigned) nd_);
            // graph_.push_back(nhood(L, S, rng, (unsigned) nd_));
        }

        const unsigned iter = parameters.Get<unsigned>("eps");
        for (unsigned n = 0; n < iter; n++) {
            // divide-and-conquer
            std::vector<unsigned> ids(nd_);
            for (unsigned i = 0; i < nd_; i++) ids[i] = i;
            Divide(parameters, ids);
            BruteForce(parameters);
            std::vector<std::vector<unsigned>>().swap(cluster_ids);
        }

        // for (unsigned i = 0; i < nd_; i++) {
        //   std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
        //   graph_[i].pool.reserve(L);
        // }
    }

    void IndexGraph::Divide(const Parameters &parameters, std::vector<unsigned> &indices) {
        if (indices.size() <= parameters.Get<unsigned>("W")) {
            cluster_ids.push_back(indices);
            return;
        }

        const unsigned C = parameters.Get<unsigned>("C");

        // randomly choose C center ids
        unsigned centers[C];
        std::mt19937 rng(rand());
        GenRandom(rng, centers, C, indices.size());
        for (unsigned i = 0; i < C; i++) centers[i] = indices[centers[i]];

        // assign each point to the nearest clusters
        std::vector<std::vector<unsigned> > clusters(C);
        std::vector<std::mutex> locks(C);
        std::vector<std::vector<Neighbor> > neighbors(indices.size());
#pragma omp parallel for
        for (unsigned i = 0; i < indices.size(); i++) {
            for (unsigned j = 0; j < C; j++) {
                float dist = distfunc(data_ + indices[i] * dimension_, data_ + centers[j] * dimension_, &dimension_);
                neighbors[i].push_back(Neighbor(j, dist, true));
            }
        }

#pragma omp parallel for
        for (unsigned i = 0; i < indices.size(); i++) {
            std::sort(neighbors[i].begin(), neighbors[i].end());
        }

        for (unsigned i = 0; i < indices.size(); i++) {
            clusters[neighbors[i][0].id].push_back(indices[i]);
        }

        for (unsigned i = 0; i < C; i++) {
            Divide(parameters, clusters[i]);
        }
    }

}
