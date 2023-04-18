//
// Created by 付聪 on 2017/6/26.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>


void load_data(char *filename, float *&data, unsigned &num, unsigned &dim) {// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    dim = 100;
    in.read((char*)&num,4);
    data = new float[num * dim * sizeof(float)];
    for (size_t i = 0; i < num; i++) {
        in.read((char *) (data + i * dim), dim * 4);
    }
    in.close();
}
void load_gt(char *filename, unsigned *&gt, unsigned sample_num, unsigned K) {
    gt = new unsigned[sample_num*K];
    std::ifstream in(filename, std::ios::binary);
    in.read((char*)gt, sample_num*K*sizeof(unsigned));
    in.close();
}
int main(int argc, char** argv){
  if(argc!=10){std::cout<< argv[0] <<" data_file init_graph save_graph K L iter S R"<<std::endl; exit(-1);}
  float* data_load = NULL;
  unsigned *gt = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  load_gt(argv[9], gt, 100000, 100);
  char* init_graph_filename = argv[2];
  char* graph_filename = argv[3];
  unsigned K = (unsigned)atoi(argv[4]);
  unsigned L = (unsigned)atoi(argv[5]);
  unsigned iter = (unsigned)atoi(argv[6]);
  unsigned S = (unsigned)atoi(argv[7]);
  unsigned R = (unsigned)atoi(argv[8]);
  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));

  index.Load(init_graph_filename);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);

  index.gt_ = gt;
  auto s = std::chrono::high_resolution_clock::now();

  index.RefineGraph(data_load, paras);

  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e-s;
  std::cout <<"Time cost: "<< diff.count() << "\n";

  index.Save(graph_filename);

  return 0;
}
