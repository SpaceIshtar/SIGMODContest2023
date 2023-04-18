//
// Created by jiarui on 4/1/23.
//

#ifndef EFANNA2E_LJR_NNDES_H
#define EFANNA2E_LJR_NNDES_H

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include "lru_cache.h"


namespace efanna2e {

    struct mutex_wrapper : std::mutex
    {
        mutex_wrapper() = default;
        mutex_wrapper(mutex_wrapper const&) noexcept : std::mutex() {}
        bool operator==(mutex_wrapper const&other) noexcept { return this==&other; }
    };

    class LJRIndexGraph : public Index {
    public:
        explicit LJRIndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer):
                    Index(dimension, n, m),initializer_{initializer}{
            assert(dimension == initializer->GetDimension());
        }

        ~LJRIndexGraph(){}

        void Save(const char *filename)override{

        }
        void Load(const char *filename)override{

        }

        void Build(size_t n, const float *data, const Parameters &parameters) override{
            data_ = data;
            timer.tick();
            InitializeGraph(parameters);
            timer.tuck("Initialize");
            eval_recall();
            NNDescent(parameters);
            final_graph_.resize(nd_);
            unsigned K = parameters.Get<unsigned>("K");
#pragma omp parallel for
            for (unsigned i = 0; i < nd_; i++) {
                std::vector<unsigned> tmp;
                std::sort(graph_[i].pool.begin(), graph_[i].pool.end());
                for (unsigned j = 0; j < K; j++) {
                    tmp.push_back(graph_[i].pool[j].id);
                }
                tmp.reserve(K);
                final_graph_[i] = tmp;
                std::vector<Neighbor>().swap(graph_[i].pool);
                std::vector<unsigned>().swap(graph_[i].nn_new);
                std::vector<unsigned>().swap(graph_[i].nn_old);
                std::vector<unsigned>().swap(graph_[i].rnn_new);
                std::vector<unsigned>().swap(graph_[i].rnn_new);
            }
            std::vector<nhood>().swap(graph_);
            has_built = true;
        }

        void Search(
                const float *query,
                const float *x,
                size_t k,
                const Parameters &parameters,
                unsigned *indices) override{

        }


        typedef std::vector<std::vector<unsigned>> CompactGraph;
        CompactGraph final_graph_;
        std::vector<mutex_wrapper> locks;

    protected:
        typedef std::vector<nhood> KNNGraph;
        Index *initializer_;
        KNNGraph graph_;



    private:
        void InitializeGraph(const Parameters &parameters){
            const unsigned L = parameters.Get<unsigned>("L");
            const unsigned S = parameters.Get<unsigned>("S");
            graph_.reserve(nd_);
            std::mt19937 rng(rand());
            for (unsigned i = 0; i < nd_; i++) {
                graph_.emplace_back(L, S, (unsigned) nd_);
            }
#pragma omp parallel for
            for (unsigned i = 0; i < nd_; i++) {
                const float *query = data_ + i * dimension_;
                std::vector<unsigned> tmp(S + 1);
                initializer_->Search(query, data_, S + 1, parameters, tmp.data());

                for (unsigned j = 0; j < S; j++) {
                    unsigned id = tmp[j];
                    if (id == i)continue;
                    float dist = distfunc(data_ + i * dimension_, data_ + id * dimension_, &dimension_);
                    graph_[i].pool.emplace_back(Neighbor(id, dist, true));
                }
                std::make_heap(graph_[i].pool.begin(), graph_[i].pool.end());
                graph_[i].pool.reserve(L);
                graph_[i].nn_new.resize(graph_[i].pool.size());
                for (unsigned j = 0; j < graph_[i].pool.size(); ++j){
                    graph_[i].nn_new[j]=graph_[i].pool[j].id;
                }
            }
        }

        void NNDescent(const Parameters &parameters){
            unsigned iter = parameters.Get<unsigned>("iter");
            unsigned S = parameters.Get<unsigned>("S");
            unsigned R = parameters.Get<unsigned>("R");
            unsigned L = parameters.Get<unsigned>("L");
            for (int it = 0; it<iter; it++){
                // if neighbor.flag = True, traverse both nn_new and nn_old
                // if neighbor.flag = False, traverse only nn_new
#pragma omp parallel for
                for (unsigned i=0;i<nd_;++i){
                    auto& nnhd = graph_[i];
                    unsigned neighbor_size = nnhd.pool.size();
                    auto& candidates = nnhd.candidates;
                    for (unsigned j = 0; j < neighbor_size; ++j){
                        auto& nn = nnhd.pool[j];
                        auto& nn_neighbor = graph_[nn.id];
                        for (unsigned c: nn_neighbor.nn_new){
                            bool check = true;
                            {
//                                LockGuard lock(nnhd.lock);
                                check = candidates.find(c) == candidates.end();
                            }
                            if (check){
                                {
//                                    LockGuard lock(nnhd.lock);
                                    candidates.emplace(c);
                                }

//                                LockGuard lock(graph_[c].lock);
//                                if (graph_[c].candidates.find(i)==graph_[c].candidates.end()){
//                                    graph_[c].candidates.emplace(i);
//                                }
                            }
                        }
                        if (nn.flag){
                            for (unsigned c: nn_neighbor.nn_old){
                                bool check = true;
                                {
//                                    LockGuard lock(nnhd.lock);
                                    check = candidates.find(c) == candidates.end();
                                }
                                if (check){
                                    {
//                                        LockGuard lock(nnhd.lock);
                                        candidates.emplace(c);
                                    }
//                                    LockGuard lock(graph_[c].lock);
//                                    if (graph_[c].candidates.find(i)==graph_[c].candidates.end()){
//                                        graph_[c].candidates.emplace(i);
//                                    }
                                }
                            }
                        }
                        nn.flag = false;
                        if (candidates.size()>=R) break;
                    }
                    if (i%1000000==0){
                        std::cout<<"ID: "<<i<<", Candidate Size: "<<candidates.size()<<std::endl;
                    }

                }
                timer.tuck("Join Ends");
#pragma omp parallel for
                for (unsigned i = 0; i < nd_; ++i){
                    auto& candidates = graph_[i].candidates;
                    unsigned count = 0;
                    for (auto iterator = candidates.begin(); iterator!=candidates.end();++iterator){
//                        unsigned id = iterator->first;
//                        float dist = iterator->second;
                        unsigned id = *iterator;
                        count++;
                        float dist = distfunc(data_ + i*dimension_, data_ + id*dimension_,&dimension_);
                        graph_[i].insert(id,dist);
                        graph_[id].insert(i,dist);
                        if (count == R) break;
                    }
                    for (auto& neighbor: graph_[i].pool){
                        if (neighbor.flag){
                            graph_[i].nn_new.emplace_back(neighbor.id);
                        }
                        else{
                            graph_[i].nn_old.emplace_back(neighbor.id);
                        }
                    }
                    candidates.clear();
                }
                timer.tuck("Update Ends");
                eval_recall();
            }


        }

        void eval_recall(){
            unsigned K = 100, sample_num = 100000;
            float mean_acc = 0;
            for (unsigned i = 0; i < sample_num; i++) {
                float acc = 0;
                auto &g1 = graph_[i].pool;
                auto g = std::vector<efanna2e::Neighbor>(g1);
                unsigned k = K < g.size() ? K : g.size();
                std::partial_sort(g.begin(), g.begin() + k, g.end());
                for (unsigned j = 0; j < k; j++) {
                    for (unsigned h = 0; h < K; h++) {
                        if (g[j].id == gt_[i * K + h]) {
                            acc++;
                            break;
                        }
                    }
                }
                mean_acc += acc / K;
            }
            std::cout << "recall : " << mean_acc / sample_num << std::endl;
        }

    };

}

#endif //EFANNA2E_LJR_NNDES_H
