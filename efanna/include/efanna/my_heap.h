//
// Created by jiarui on 4/4/23.
//

#ifndef EFANNA2E_MY_HEAP_H
#define EFANNA2E_MY_HEAP_H
#include <vector>
#include <algorithm>
namespace efanna2e {
    template<typename T>
    class heap {
    public:
        T &operator[](unsigned i) {
            return data[i];
        }

        std::size_t size() {
            return data.size();
        }

        void make_heap() {
            std::make_heap(data.begin(), data.end());
        }

        void sort() {
            std::sort(data.begin(), data.end());
        }

        T &back() {
            return data.back();
        }

        T &front() {
            return data.front();
        }

        void reserve(std::size_t R){
            data.reserve(R);
        }

        std::size_t capacity() const{
            return data.capacity();
        }

        void insert(T t) {
            data.push_back(t);
            unsigned pos = data.size() - 1;
            while (pos > 1) {
                unsigned next_pos = (pos - 1) / 2;
                if (data[pos] > data[next_pos]) {
                    T tmp = data[pos];
                    data[pos] = data[next_pos];
                    data[next_pos] = tmp;
                    pos = next_pos;
                } else {
                    break;
                }
            }
        }
        void push_back(T t){
            data.push_back(t);
        }
        void pop_insert(T t) {
            data[0] = t;
            unsigned pos = 0;
            unsigned size = data.size();
            while (true) {
                unsigned left = pos * 2 + 1;
                unsigned right = left + 1;
                if (right < size) {
                    if (data[left] < data[right]) {
                        if (data[pos] < data[right]) {
                            T tmp = data[pos];
                            data[pos] = data[right];
                            data[right] = tmp;
                            pos = right;
                        } else {
                            break;
                        }
                    } else {
                        if (data[pos] < data[left]) {
                            T tmp = data[pos];
                            data[pos] = data[left];
                            data[left] = tmp;
                            pos = left;
                        } else {
                            break;
                        }
                    }
                } else {
                    if (left < size) {
                        if (data[pos] < data[left]) break;
                        else {
                            T tmp = data[pos];
                            data[pos] = data[left];
                            data[left] = tmp;
                            pos = left;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

    std::vector<T> data;
    };
}
#endif //EFANNA2E_MY_HEAP_H
