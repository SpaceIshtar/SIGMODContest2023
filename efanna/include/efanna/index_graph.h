//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_GRAPH_H
#define EFANNA2E_INDEX_GRAPH_H

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"

namespace efanna2e {

    struct mutex_wrapper : std::mutex
    {
        mutex_wrapper() = default;
        mutex_wrapper(mutex_wrapper const&) noexcept : std::mutex() {}
        bool operator==(mutex_wrapper const&other) noexcept { return this==&other; }
    };

class IndexGraph : public Index {
 public:
  explicit IndexGraph(const size_t dimension, const size_t n, Metric m, Index *initializer);

  explicit IndexGraph(const size_t dimension, const size_t n, Metric m, const Parameters &param);

  virtual ~IndexGraph();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;

  void RefineBySearch(const Parameters &parameters);
  void RefineByBFS(const Parameters &parameters);
  void RefineByHubness(const Parameters &parameters);
  void BruteForce(const Parameters &parameters);
  void RandomMultipleDivision(const Parameters &parameters);
  void Divide(const Parameters &parameters, std::vector<unsigned> &indices);
  void eval_final_recall();
  void mergePool(IndexGraph* index);
  void eval_recall();
  void join_with_filter();
  void join_batch();
  void NNDescentWithFilter(const Parameters &parameters);
  void NNDescent_batch(const Parameters &parameters);
  void Build_batch(size_t n, const float *data, const Parameters &parameters);

  std::vector<std::vector<unsigned>> cluster_ids;
  std::vector<std::vector<unsigned>> visited_list;
  unsigned* final_graph_;
  // typedef std::vector<std::vector<unsigned>> CompactGraph;
  // CompactGraph final_graph_;
  unsigned init_seed = 1234;
  std::size_t pool_len_;


 protected:
  typedef std::vector<nhood> KNNGraph;
  typedef std::vector<LockNeighbor> LockGraph;

  Index *initializer_;
  KNNGraph graph_;

//   std::vector<std::vector<std::pair<float, unsigned>>> dist_pool_;
  std::vector<std::vector<DistId>> dist_pool_;
  std::size_t thread_pool_size_;


 private:
  void InitializeGraph(const Parameters &parameters);
  void InitializeGraph_Refine(const Parameters &parameters);
  void NNDescent(const Parameters &parameters);
  void join();
  void update(unsigned L, unsigned S, unsigned R, unsigned iteration);
  void join_sorted();
  void update_sorted(unsigned L,unsigned S, unsigned R, unsigned iteration);
  void update_guodu(unsigned L, unsigned S, unsigned R, unsigned iteration);
  void generate_control_set(std::vector<unsigned> &c,
                                      std::vector<std::vector<unsigned> > &v,
                                      unsigned N);

  void get_neighbor_to_add(const float* point, const Parameters &parameters, LockGraph& g,
                           std::mt19937& rng, std::vector<Neighbor>& retset, unsigned n_total);
  void compact_to_Lockgraph(LockGraph &g);
  void parallel_graph_insert(unsigned id, Neighbor nn, LockGraph& g, size_t K);

};

}

#endif //EFANNA2E_INDEX_GRAPH_H
