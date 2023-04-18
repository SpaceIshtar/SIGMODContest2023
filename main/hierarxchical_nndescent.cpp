//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <efanna2e/distfunc.h>
#include <omp.h>
#include <chrono>
#include <cstring>
#include <unordered_set>


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

float eval_recall(std::vector<std::vector<unsigned>>& res, unsigned* gt) {
  unsigned K = 100, sample_num = 100000;
  unsigned acc_count = 0;
  for(unsigned i=0; i<sample_num; i++){
    for (unsigned k=0;k<K;++k){
        for (unsigned p=0;p<res[i].size();++p){
            if (res[i][p]==gt[i*K+k]){
                acc_count++;
                break;
            }
        }
//        acc_count += res[i].count(gt[i*K+k]);
    }
  }
  return (float)acc_count/(float)(K*sample_num);
}

int main(int argc, char **argv) {
    omp_set_num_threads(32);

    if (argc != 10) {
        std::cout << argv[0] << " data_file K L iter S R gt cluster_num replica_max" << std::endl;
        exit(-1);
    }
    float *data_load = NULL;
    unsigned *gt = NULL;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);
    load_gt(argv[7], gt, 100000, 100);
    unsigned K = (unsigned) atoi(argv[2]);
    unsigned L = (unsigned) atoi(argv[3]);
    unsigned iter = (unsigned) atoi(argv[4]);
    unsigned S = (unsigned) atoi(argv[5]);
    unsigned R = (unsigned) atoi(argv[6]);
    unsigned cluster_num = (unsigned) atoi(argv[8]);
    unsigned replica_count = (unsigned) atoi(argv[9]);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    assert((points_num%cluster_num)==0);
    assert((points_num/cluster_num)%10==0);

    auto partition_start = std::chrono::high_resolution_clock::now();
    std::cout<<"Partition start"<<std::endl;
    // partition into clusters，每个cluster设置10000个点，一共1000个cluster
    // 直接按id顺序分cluster，然后每个cluster从其余cluster当中每个都随机选取p个点添加进来，这样每个cluster就有将近12000个点
    std::vector<std::vector<unsigned>> partitions(cluster_num,std::vector<unsigned>());
    unsigned cluster_base_size = points_num/cluster_num;
    
    std::mt19937 mt_rand(std::chrono::system_clock::now().time_since_epoch().count());
    #pragma omp parallel for
    for (unsigned i=0;i<cluster_num;++i){
        unsigned base_id = i*cluster_base_size;
        for (unsigned j=0;j<cluster_base_size;++j){
            partitions[i].push_back(base_id + j);
        }
        // 每个cluster随机选p个点做replica
        for (unsigned j=0;j<cluster_num;++j){
            if (i==j) continue;
            unsigned replica_id = mt_rand() % cluster_base_size;
            replica_id = j*cluster_base_size+replica_id;
            for (unsigned p=0;p<replica_count;++p){
                partitions[i].push_back(replica_id);
                replica_id = (replica_id + 1) % cluster_base_size;
            }
        }
    }

    auto partition_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = partition_end - partition_start;
    std::cout<<"Partition end. Time cost: "<<diff.count()<<std::endl;

    std::vector<std::vector<unsigned>> graph(points_num,std::vector<unsigned>());
    for (unsigned i=0;i<cluster_num;++i){
        unsigned cluster_size = partitions[i].size();
        float* cluster_data = new float[cluster_size*dim];
        #pragma omp parallel for
        for (unsigned j=0;j<cluster_size;++j){
            memcpy(cluster_data+j*dim, data_load+partitions[i][j]*dim, dim*sizeof(float));
        }
//        std::cout<<"After memcpy"<<std::endl;
        efanna2e::IndexRandom init_index(dim, cluster_size);
        efanna2e::IndexGraph index(dim, cluster_size, efanna2e::L2, (efanna2e::Index *) (&init_index));
        index.distfunc = &utils::L2SqrFloatAVX512;
        index.Build(cluster_size, cluster_data, paras);
//        std::cout<<"After Build"<<std::endl;
        auto final_graph = index.final_graph_;
//        #pragma omp parallel for
        for (unsigned j=0;j<cluster_size;++j){
            auto neighbors = final_graph[j];
            for (unsigned neighbor_sub_id: neighbors){
//                assert(neighbor_sub_id<cluster_size);
//                assert(partitions[i][j]<points_num);
//                assert(partitions[i][neighbor_sub_id]<points_num);
                if (std::find(graph[partitions[i][j]].begin(), graph[partitions[i][j]].end(),partitions[i][neighbor_sub_id])==graph[partitions[i][j]].end()){
                    graph[partitions[i][j]].emplace_back(partitions[i][neighbor_sub_id]);
                }
            }
        }
        delete[] cluster_data;
    } 

    auto nnd_end = std::chrono::high_resolution_clock::now();
    diff = nnd_end - partition_end;
    std::cout<<"nndescent ends. Time cost: "<<diff.count()<<std::endl;

    float recall = eval_recall(graph,gt);
    std::cout<<"Recall = "<<recall<<std::endl;

    return 0;
}
