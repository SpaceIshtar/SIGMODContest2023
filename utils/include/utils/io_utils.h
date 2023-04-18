//
// Created by longxiang on 3/15/23.
//

#ifndef SIGMOD_CONTEST_IO_UTILS_H
#define SIGMOD_CONTEST_IO_UTILS_H

#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNNG(const std::vector<std::vector<uint32_t>> &knng,
              const std::string &path);

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             std::vector<std::vector<float>> &data);

template<typename T>
T* load_vecs(const std::string &kFilePath, size_t &vecs_num, size_t &vecs_dim) {
    std::ifstream fin(kFilePath, std::ios::binary);    //以二进制的方式打开文件
    if (!fin.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    unsigned int dim;
    fin.read((char *) &dim, 4);    //读取向量维度
    vecs_dim = (size_t) dim;
    fin.seekg(0, std::ios::end);    //光标定位到文件末尾
    std::ios::pos_type ss = fin.tellg();    //获取文件大小（多少字节）
    auto fsize = (size_t) ss;
    vecs_num = fsize / (vecs_dim + 1) / 4;    //数据的个数
    T* data = new T[vecs_num * vecs_dim];
//    std::unique_ptr<T[]> get_data = std::make_unique<T[]>(vecs_num * vecs_dim);

    fin.seekg(0, std::ios::beg);    //光标定位到起始处
    for (size_t i = 0; i < vecs_num; i++) {
        fin.seekg(4, std::ios::cur);    //光标向右移动4个字节
        fin.read((char *) (data + i * vecs_dim), vecs_dim * 4);    //读取数据到一维数据data中
    }

    fin.close();
    return data;
}

void FastRead(const std::string &file_path, float *&data, std::size_t &num);

void ReadGt(const std::string &kFilePath, unsigned *&gt, const std::size_t kNum, const std::size_t kTopK);

template<typename T>
void WriteData(const std::string &file_path, T *&data, uint32_t num) {
    FILE *fp = nullptr;
    fp = fopen(file_path.c_str(), "wb");
    assert(fp!=nullptr);
    fwrite(&num, sizeof(uint32_t), 1, fp);

    fwrite(data, sizeof(T), num*100, fp);
    fclose(fp);
}

template<typename T>
void WriteData(const std::string &kFilePath, T *&data, const std::size_t num, const std::size_t dim) {
    FILE *fp = nullptr;
    fp = fopen(kFilePath.c_str(), "wb");
    assert(fp!=nullptr);

    fwrite(data, sizeof(T), num*dim, fp);
    fclose(fp);
}

void load_data(char *filename, float *&data, unsigned &num, unsigned &dim);

#endif //SIGMOD_CONTEST_IO_UTILS_H
