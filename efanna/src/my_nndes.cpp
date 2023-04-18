#include <efanna/my_nndes.h>
#include <efanna/exceptions.h>
#include <efanna/parameters.h>
#include <omp.h>
#include <set>
#include <cstring>
#include <algorithm>

namespace efanna2e {
MyIndex::MyIndex(const size_t dimension, const size_t n, Metric m) : Index(dimension, n, m) {}
MyIndex::~MyIndex() {}

void MyIndex::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
  for (unsigned n = 0; n < nd_; n++) {
    graph_[n].join([&](unsigned i, unsigned j) {
      if(i != j){
        float dist = distfunc(data_ + i * dimension_, data_ + j * dimension_, &dimension_);
        graph_[i].insert(j, dist);
        graph_[j].insert(i, dist);
      }
    });
  }
}

void MyIndex::update(const Parameters &parameters) {
  unsigned S = parameters.Get<unsigned>("S");
  unsigned L = parameters.Get<unsigned>("L");
#pragma omp parallel for
  for (unsigned i = 0; i < nd_; i++) {
    std::vector<unsigned>().swap(graph_[i].nn_new);
    std::vector<unsigned>().swap(graph_[i].nn_old);
  }
#pragma omp parallel for
  for (unsigned n = 0; n < nd_; ++n) {
    auto &nn = graph_[n];
    std::sort(nn.pool.begin(), nn.pool.end());
    unsigned maxl = std::min(nn.M + S, (unsigned) nn.pool.size());
    unsigned c = 0;
    unsigned l = 0;
    while ((l < maxl) && (c < S)) {
      if (nn.pool[l].flag) ++c;
      ++l;
    }
    nn.M = l;
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
        nn.flag = false;
      } else {
        nn_old.push_back(nn.id);
      }
    }
    std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
  }
}

void MyIndex::NNDescent(const Parameters &parameters) {
  unsigned iter = parameters.Get<unsigned>("iter");
  for (unsigned it = 0; it < iter; it++) {
    std::cout << "iter: ";
    update(parameters);
    join();
    timer.tuck(std::to_string(it));
    eval_recall();
  }
}

void MyIndex::eval_recall() {
  unsigned K = 100, sample_num = 100000;
  float mean_acc = 0;
  for(unsigned i=0; i<sample_num; i++){
    float acc = 0;
    auto &g1 = graph_[i].pool;
    auto g = std::vector<efanna2e::Neighbor>(g1);
    unsigned k = K < g.size() ? K : g.size();
    std::partial_sort(g.begin(), g.begin() + k, g.end());
    for(unsigned j=0; j<k; j++){
      for(unsigned h=0; h<K; h++){
        if(g[j].id == gt_[i*K+h]){
          acc++;
          break;
        }
      }
    }
    mean_acc += acc / K;
  }
  std::cout<<"recall : "<< mean_acc / sample_num << std::endl;
}

void MyIndex::Build(size_t n, const float *data, const Parameters &parameters) {
  data_ = data;

  RandomMultipleDivision(parameters);
  timer.tuck("Init");
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
  }
  std::vector<my_nhood>().swap(graph_);
  has_built = true;
}

void MyIndex::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);
  unsigned GK = (unsigned) final_graph_[0].size();
  for (unsigned i = 0; i < nd_; i++) {
    out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void MyIndex::BruteForce(const Parameters &parameters) {
  const unsigned W = parameters.Get<unsigned>("W");
  const unsigned L = parameters.Get<unsigned>("L");
  const unsigned clusters_num = cluster_ids.size();
  
  for (unsigned cls = 0; cls < clusters_num; cls++) {
    Neighbor *neighbors = new Neighbor[W*W];
    auto indices = cluster_ids[cls];
    const unsigned w = indices.size();
    #pragma omp parallel for
    for (unsigned i = 0; i < w - 1; i++) {
      for (unsigned j = i + 1; j < w; j++) {
        float dist = distfunc(data_ + indices[i]*dimension_, data_ + indices[j]*dimension_, &dimension_);
        neighbors[i*W+j] = Neighbor(indices[j], dist, true);
        neighbors[j*W+i] = Neighbor(indices[i], dist, true);
      }
    }

    #pragma omp parallel for
    for (unsigned i = 0; i < w; i++) {
      for (unsigned j = 0; j < w; j++) {
        auto &pool = graph_[indices[i]].pool;
        if (std::find(pool.begin(), pool.end(), neighbors[i*W+j]) == pool.end() && i != j) {
          if (pool.size() < L) {
            pool.push_back(neighbors[i*W+j]);
            std::push_heap(pool.begin(), pool.end());
          } else if (neighbors[i*W+j].distance < pool.front().distance) {
            std::pop_heap(pool.begin(), pool.end());
            pool[pool.size()-1] = neighbors[i*W+j];
            std::push_heap(pool.begin(), pool.end());
          }
        }
      }
    }

    delete[] neighbors;
  }
}

void MyIndex::RandomMultipleDivision(const Parameters &parameters) {
  const unsigned L = parameters.Get<unsigned>("L");
  const unsigned S = parameters.Get<unsigned>("S");

  // initialization
  graph_.resize(nd_);
  std::mt19937 rng(rand());
#pragma omp parallel for
  for (unsigned i = 0; i < nd_; i++) {
    graph_[i] = my_nhood(L, S);
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
}

void MyIndex::Divide(const Parameters &parameters, std::vector<unsigned> &indices) {
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
      float dist = distfunc(data_ + indices[i]*dimension_, data_ + centers[j]*dimension_, &dimension_);
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
