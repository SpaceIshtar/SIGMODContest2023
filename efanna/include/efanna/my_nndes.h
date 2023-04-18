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

class MyIndex : public Index {
 public:
  explicit MyIndex(const size_t dimension, const size_t n, Metric m);

  virtual ~MyIndex();

  virtual void Save(const char *filename)override;

  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Load(const char *filename)override {};

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override {};

  void BruteForce(const Parameters &parameters);
  void RandomMultipleDivision(const Parameters &parameters);
  void Divide(const Parameters &parameters, std::vector<unsigned> &indices);

  std::vector<std::vector<unsigned> > visited_list;
  std::vector<std::vector<unsigned> > cluster_ids;

 protected:
  typedef std::vector<my_nhood> KNNGraph;
  typedef std::vector<std::vector<unsigned > > CompactGraph;
  typedef std::vector<LockNeighbor > LockGraph;

  Index *initializer_;
  KNNGraph graph_;
  CompactGraph final_graph_;

 private:
  void NNDescent(const Parameters &parameters);
  void join();
  void update(const Parameters &parameters);
  void eval_recall();
};

}

#endif //EFANNA2E_INDEX_GRAPH_H
