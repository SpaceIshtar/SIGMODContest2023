//
// Created by jiarui on 3/22/23.
//

#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <efanna2e/distfunc.h>
#include <omp.h>
#include <chrono>
#include <cstring>
#include <unordered_set>
#include <queue>

struct mutex_wrapper : std::mutex
{
    mutex_wrapper() = default;
    mutex_wrapper(mutex_wrapper const&) noexcept : std::mutex() {}
    bool operator==(mutex_wrapper const&other) noexcept { return this==&other; }
};

struct smallHeapFunc{
    bool operator()(std::pair<float,unsigned> a, std::pair<float,unsigned> b){
        return a.first>b.first;
    }};

struct largeHeapFunc{
    bool operator()(std::pair<float,unsigned> a, std::pair<float,unsigned> b){
        return a.first<b.first;
    }};


void load_centers(std::string& filename, float* centers, unsigned center_num, unsigned dim){
    std::ifstream csv_data(filename);
//    centers=new float[center_num*dim];
    std::string line;
    std::istringstream sin;
    std::string word;
    unsigned line_id=0;
    while (std::getline(csv_data, line)){
        sin.clear();
        sin.str(line);
        unsigned col_id=0;
        while (std::getline(sin, word, ',')){
            centers[line_id*dim+col_id]=atof(word.c_str());
            col_id++;
        }
        line_id++;
    }
    csv_data.close();
}

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

float eval_recall(std::vector<std::priority_queue<std::pair<float, unsigned>,std::vector<std::pair<float,unsigned>>,largeHeapFunc>>& knn_res, unsigned* gt){
    unsigned acc_count=0;
    unsigned total_num=100000*100;
//    std::ofstream os("/home/jiaruiluo/Sigmod2023/SIGMOD_Contest/kmeans/hierarchy_nndescnet_result100k_L128_R150.csv");
    for (unsigned i=0;i<100000;i++){
        std::unordered_set<unsigned> set;
        for (unsigned j=0;j<100;j++){
            set.insert(gt[i*100+j]);
        }
        while (!knn_res[i].empty()){
            std::pair<float, unsigned> top_val=knn_res[i].top();
            //    os<<top_val.second;
            acc_count+=set.count(top_val.second);
            knn_res[i].pop();
            //    if (knn_res[i].empty()){
            //        os<<"\n";
            //    }
            //    else{
            //        os<<",";
            //    }
        }
    }
//    os.close();
    return (float) acc_count/(float) total_num;
}

int main(int argc, char **argv) {
    omp_set_num_threads(32);

    if (argc != 11) {
        std::cout << argv[0] << " data_file K L iter S R gt center_num eps replica_max" << std::endl;
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
    unsigned center_num = (unsigned) atoi(argv[8]);
    float eps = atof(argv[9]);
    unsigned replica_max = (unsigned) atoi(argv[10]);
    std::size_t dim_sizet = dim;

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    float* centers = new float[center_num*dim];
    std::string center_path = "/home/jiarui/hierarchical_nndescent/SIGMOD_Contest/centers/kmeans_"+std::to_string(center_num)+"centers.csv";
    std::cout<<"Load centers from: "<<center_path<<std::endl;
    std::cout<<"Points num: "<<points_num<<", Centers num: "<<center_num<<std::endl;
    load_centers(center_path,centers,center_num,dim);

    auto start_time=std::chrono::high_resolution_clock::now();
    std::vector<std::vector<unsigned>> cluster2id(center_num,std::vector<unsigned>());
    std::vector<mutex_wrapper> locks(center_num,mutex_wrapper());

#pragma omp parallel for
    for (unsigned i=0;i<points_num;++i){
        std::priority_queue<std::pair<float,unsigned>,std::vector<std::pair<float,unsigned>>,smallHeapFunc> pq;
        for (unsigned j=0;j<center_num;j++){
            float dis=utils::L2SqrFloatAVX512(data_load+i*dim,centers+j*dim,&dim_sizet);
            pq.emplace(dis,j);
        }
        std::pair<float, unsigned> current=pq.top();
        float best=current.first*(1+eps);
        unsigned replica_num=0;
        while (current.first<best){
            unsigned cid=current.second;
            locks[cid].lock();
            cluster2id[cid].push_back(i);
            locks[cid].unlock();
            pq.pop();
//            id2cluster[i].emplace(cid);
            if (pq.empty()) break;
            current=pq.top();
            replica_num++;
            if (replica_num==replica_max) break;
        }
    }

    auto allocate_center_end=std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = allocate_center_end - start_time;
    std::cout<<"Allocating Centers ends, time cost: "<<diff.count()<<std::endl;

    std::cout<<std::endl;
    for (unsigned i=0;i<center_num;i++){
        std::cout<<"Cluster "<<i<<" has size: "<<cluster2id[i].size()<<std::endl;
    }
    std::cout<<std::endl;

    std::vector<std::unordered_set<unsigned>> candidate_pools(points_num,std::unordered_set<unsigned>());
    for (unsigned i=0;i<center_num;i++){
        auto index_start=std::chrono::high_resolution_clock::now();
//        faiss::IndexNNDescentFlat index(dim,100);
//        index.nndescent.S=50; index.nndescent.L=175; index.nndescent.R=300;
//        index.nndescent.iter=2;
        unsigned cluster_size=cluster2id[i].size();
        float* cluster_data=new float[cluster_size*dim];
#pragma omp parallel for
        for (unsigned j=0;j<cluster_size;++j){
            memcpy(cluster_data+j*dim,data_load+cluster2id[i][j]*dim,dim*sizeof(float));
        }
        //    cluster_data=efanna2e::data_align(cluster_data, cluster_size, dim);
        auto mem_copy_end=std::chrono::high_resolution_clock::now();
        diff = mem_copy_end - index_start;
        std::cout<<"Mem cpy for index "<<i<<" requires time cost: "<<diff.count()<<std::endl;
//        index.add(cluster_size,cluster_data);

        efanna2e::IndexRandom init_index(dim, cluster_size);
        efanna2e::IndexGraph index(dim, cluster_size, efanna2e::L2, (efanna2e::Index *) (&init_index));
        index.distfunc = &utils::L2SqrFloatAVX512;
        index.Build(cluster_size, cluster_data, paras);
        auto index_build=std::chrono::high_resolution_clock::now();
        diff=index_build-mem_copy_end;
        std::cout<<"Building index "<<i<<" requires time cost: "<<diff.count()<<std::endl;
#pragma omp parallel for
        for (unsigned j=0;j<cluster_size;++j){
            auto knn_graph=index.final_graph_[j];
            unsigned self_id=cluster2id[i][j];
            unsigned neighbor_size=knn_graph.size();
            for (unsigned k=0;k<neighbor_size;k++){
                unsigned neigh_id=cluster2id[i][knn_graph[k]];
                if (self_id==neigh_id) continue;
                if (candidate_pools[self_id].count(neigh_id)) continue;
                candidate_pools[self_id].emplace(neigh_id);
            }
        }
        delete[] cluster_data;
        auto build_knn_graph=std::chrono::high_resolution_clock::now();
        diff = build_knn_graph - index_build;
        std::cout<<"Building knn graph for cluster "<<i<<" requires time cost: "<<diff.count()<<std::endl;
    }

    auto before_dis_computation=std::chrono::high_resolution_clock::now();
    std::vector<std::priority_queue<std::pair<float,unsigned>, std::vector<std::pair<float,unsigned>>,largeHeapFunc>> knn_res(points_num,std::priority_queue<std::pair<float,unsigned>, std::vector<std::pair<float,unsigned>>,largeHeapFunc>());
#pragma omp parallel for
    for (unsigned i=0;i<points_num;++i){
        auto candidates=candidate_pools[i];
        if (candidates.size()<=K){
            for (auto iter=candidates.begin();iter!=candidates.end();++iter){
                knn_res[i].emplace(0,*iter);
            }
        }
        else{
            for (auto iter=candidates.begin();iter!=candidates.end();++iter){
                float dis=utils::L2SqrFloatAVX512(data_load+i*dim,data_load+(*iter)*dim,&dim_sizet);
                knn_res[i].emplace(dis,*iter);
                if (knn_res[i].size()>K) knn_res[i].pop();
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - before_dis_computation;
    std::cout<<"Distance Computation Requires time cost: "<<diff.count()<<std::endl;
    std::cout<<std::endl;
    diff = end_time - start_time;
    std::cout<<"The total time cost: "<<diff.count()<<std::endl;

    float recall = eval_recall(knn_res, gt);
    std::cout<<"The final recall is "<<recall<<std::endl;

    delete[] centers;
}