//
// Created by 付聪 on 2017/6/21.
//

#include <efanna/index_graph.h>
#include <efanna/index_random.h>
#include <efanna/util.h>
#include <efanna/distfunc.h>
#include <omp.h>


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

int main(int argc, char **argv) {
    // omp_set_num_threads(32);

    // if (argc != 9) {
    //     std::cout << argv[0] << " data_file save_graph K L iter S R gt" << std::endl;
    //     exit(-1);
    // }
    float *data_load = NULL;
   unsigned *gt = NULL;
//    unsigned *gt2 = NULL;
    unsigned points_num, dim_useless;
    load_data(argv[1], data_load, points_num, dim_useless);
    // load_gt(argv[8], gt, 100000, 100);
//    load_gt(argv[9],gt2,100000,100);
    char *graph_filename = argv[2];
    unsigned K = (unsigned) atoi(argv[3]);
    unsigned L = (unsigned) atoi(argv[4]);
    unsigned iter = (unsigned) atoi(argv[5]);
    unsigned S = (unsigned) atoi(argv[6]);
    unsigned R = (unsigned) atoi(argv[7]);
    //unsigned C = (unsigned) atoi(argv[9]);
    //unsigned W = (unsigned) atoi(argv[10]);
    //unsigned eps = (unsigned) atoi(argv[11]);
    const unsigned dim = 100;
    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index *) (&init_index));
    index.distfunc = &utils::L2SqrFloatAVX512;
    efanna2e::Timer &timer = index.timer;

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

   index.gt_ = gt;
//    index.gt2_ = gt2;
    timer.tick();
    index.Build(points_num, data_load, paras);
    timer.tuck("Build nnd1");
    index.Save(graph_filename);
    timer.tuck("Save ENDS");

//    index.Load(graph_filename);
//    unsigned sample_id = 1000;
//    std::cout<<"Final Graph has size: "<<index.final_graph_.size()<<std::endl;
//    std::cout<<"Neighbor Size: "<<index.final_graph_[sample_id].size()<<std::endl;
//    for (unsigned neighbor:index.final_graph_[sample_id]){
//        float dist = utils::L2SqrFloatAVX512(data_load + sample_id * dim, data_load + neighbor * dim, &dim);
//        std::cout<<"id: "<<sample_id<<", dist: "<<dist<<std::endl;
//    }

//    efanna2e::IndexRandom init_index2(dim,points_num);
//    efanna2e::IndexGraph index2(dim,points_num,efanna2e::L2,(efanna2e::Index *) (&init_index2));
//    index2.distfunc = &utils::L2SqrFloatAVX512;
//    index2.timer = index.timer;
//    index2.gt_ = gt;
//    index2.gt2_ = gt2;
//    index2.Build(points_num, data_load, paras);
//    timer.tuck("Build nnd2");
//    index.mergePool(&index2);
//    timer.tuck("Merge Pool Ends");
//    index.eval_recall();


    return 0;
}
