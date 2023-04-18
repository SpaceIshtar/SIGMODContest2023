//
// Created by longxiang on 3/15/23.
//

#include "utils/io_utils.h"

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNNG(const std::vector<std::vector<uint32_t>> &knng,
              const std::string &path = "output.bin") {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
    const uint32_t N = knng.size();
    std::cout << "Saving KNN Graph (" << knng.size() << " X 100) to " << path
              << std::endl;
    assert(knng.front().size() == K);
    for (unsigned i = 0; i < knng.size(); ++i) {
        auto const &knn = knng[i];
        ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             std::vector<std::vector<float>> &data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points

    ifs.read((char *)&N, sizeof(uint32_t));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;

    const int num_dimensions = 100;
    std::vector<float> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float))) {
        std::vector<float> row(num_dimensions);
#pragma omp simd
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}

void ReadBin(const std::string &file_path, float *data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;

    ifs.read((char *)&N, sizeof(uint32_t));
    std::cout << "# of points: " << N << std::endl;

    const int num_dimensions =100;
    data = new float[N * num_dimensions];
    for (std::size_t i = 0; i < N; ++i) {
        ifs.read((char *)(data + i * num_dimensions), num_dimensions * sizeof(float));
    }

    ifs.close();
}

void FastRead(const std::string &file_path, float *&data, std::size_t &num) {
    FILE *fp = nullptr;
    fp = fopen(file_path.c_str(), "rb");
    assert(fp!=nullptr);
    uint32_t  tmp_num;
    fread(&tmp_num, sizeof(uint32_t), 1, fp);
    num = (std::size_t) tmp_num;
    std::cout << "Num: " << num << std::endl;

    data = new float [num*100];
    fread(data, sizeof(float), num*100, fp);
//    fread((char*) data, 1, 4*num*100, fp);
    fclose(fp);
}

void ReadGt(const std::string &kFilePath, unsigned *&gt, const std::size_t kNum, const std::size_t kTopK) {
    FILE *fp = nullptr;
    fp = fopen(kFilePath.c_str(), "rb");
    assert(fp!= nullptr);
    gt = new unsigned [kNum * kTopK];
    fread(gt, sizeof(unsigned ), kNum*kTopK, fp);
    fclose(fp);
}

void load_data(char *filename, float *&data, unsigned &num, unsigned &dim) {
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