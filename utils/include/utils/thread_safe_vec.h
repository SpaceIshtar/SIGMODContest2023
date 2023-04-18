//
// Created by longxiang on 3/21/23.
//

#ifndef SIGMOD_CONTEST_THREAD_SAFE_VEC_H
#define SIGMOD_CONTEST_THREAD_SAFE_VEC_H

#include <vector>
//#include <thread>
#include <mutex>

namespace utils{
template <typename  T>
class ThreadSafeVector{
    std::vector<T> mvec_;
    std::mutex lock_;
public:
    thread_local static std::vector<T> vec_;

    ThreadSafeVector() = default;
    ThreadSafeVector(const ThreadSafeVector& vec) {
        mvec_ = vec;
        vec_ = vec;
    }
    ThreadSafeVector(ThreadSafeVector&& vec) {
        mvec_ = vec;
        vec_ = vec;
    }

    void push_back(const T &value) noexcept {
        vec_.push_back(value);
    }

    void merge() {
        std::lock_guard<std::mutex> lockGuard(lock_);
        mvec_.insert(mvec_.end(), vec_.begin(), vec_.end());
        vec_.clear();
    }

    void get_vector(std::vector<T>& res) {
        res = mvec_;
    }
};

template<typename T>
thread_local std::vector<T> ThreadSafeVector<T>::vec_(0);

} // namespace utils
#endif //SIGMOD_CONTEST_THREAD_SAFE_VEC_H
