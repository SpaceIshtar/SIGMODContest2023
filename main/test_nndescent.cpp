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
    unsigned points_num, dim_useless;
    load_data(argv[1], data_load, points_num, dim_useless);
    // load_gt(argv[8], gt, 100000, 100);
    char *graph_filename = argv[2];
    unsigned K = (unsigned) atoi(argv[3]);
    unsigned L = (unsigned) atoi(argv[4]);
    unsigned iter = (unsigned) atoi(argv[5]);
    unsigned S = (unsigned) atoi(argv[6]);
    unsigned R = (unsigned) atoi(argv[7]);
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
    timer.tick();
    index.Build(points_num, data_load, paras);
    timer.tuck("Build nnd1");
    index.Save(graph_filename);
    timer.tuck("Save ENDS");

    return 0;
}
