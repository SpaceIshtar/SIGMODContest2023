#include "cuckoofilter/cuckoofilter.h"
#include "cuckoofilter/cuckoofilter_stable.h"
#include "cuckoofilter/simd-block-fixed-fpp.h"
// #include "xorfilter.h"
#include <chrono>
#include <random>

unsigned random_uint(unsigned int min, unsigned int max) {
    // static std::mt19937 rng(seed);
    // return (rng() % max) + min;
    thread_local std::mt19937 mt(std::random_device{}());
    std::uniform_int_distribution<unsigned> dis(min, max);
    return dis(mt);
}

template <typename T>
class Timer {
    std::chrono::steady_clock::time_point time_begin_;

public:
    Timer() = default;

    inline T getElapsedTime() {
        auto time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<T>(time_end - time_begin_);
    }

    inline void reset() {
        time_begin_ = std::chrono::steady_clock::now();
    }

    inline void tuck(std::string msg) {
       T time = getElapsedTime();
       std::cout << msg << " Time cost: " << time.count() << std::endl;
    }
};


int main() {
    Timer<std::chrono::microseconds> timer;

    std::vector<unsigned> pool(2000);

    for (unsigned i=0; i<2000; ++i) {
        pool[i] = i;
    }

    // std::vector<unsigned> rand(2000);
    // for (unsigned i=0; i<2000; ++i) {
    //     rand[i] = random_uint(0, 4000);
    // }
    unsigned rand = random_uint(0, 4000);

    timer.reset();
    // for (unsigned i=0; i<2000; ++i) {
        for (unsigned j=0; j<2000; ++j) {
            if (rand == pool[j]) break;
        }
    // }
    std::cout << "For loop time: " << timer.getElapsedTime().count() << std::endl;


    timer.reset();
    cuckoofilter::CuckooFilter<unsigned, 12> cu_filter(2000); 

    for (unsigned i=0; i<2000; ++i) {
        if (cu_filter.Add(i) != cuckoofilter::Ok) {
            std::cout << "Faile " << std::endl;
        }
    }
    std::cout << "Cuckoofilter 2k Add time: " << timer.getElapsedTime().count() << std::endl;

    timer.reset();
    unsigned false_pos = 0;
    for (unsigned i=2000; i<4000; ++i) {
        if (cu_filter.Contain(i)==cuckoofilter::Ok) {
            ++false_pos;
        }
    }
    std::cout << "Cuckoofilter Check time: " << timer.getElapsedTime().count() << std::endl;
    std::cout << "cuckoo filter false_pos: " << false_pos << std::endl;

    timer.reset();
    cuckoofilter::CuckooFilterStable<unsigned, 12> cu_filter_st(2000);
    for (unsigned i=0; i<2000; ++i) {
        if (cu_filter_st.Add(i) != cuckoofilter::Ok) {
            std::cout << "Faile " << std::endl;
        }
    }
    std::cout << "CuckoofilterStable 2k Add time: " << timer.getElapsedTime().count() << std::endl;

    timer.reset();
    false_pos = 0;
    for (unsigned i=2000; i<4000; ++i) {
        if (cu_filter_st.Contain(i)==cuckoofilter::Ok) {
            ++false_pos;
        }
    }
    std::cout << "CuckoofilterStable Check time: " << timer.getElapsedTime().count() << std::endl;
    std::cout << "CuckoofilterStable false_pos: " << false_pos << std::endl;


    timer.reset();
    SimdBlockFilterFixed64 simd_b_f(3000);
    for (unsigned i=0; i<2000; ++i) {
        simd_b_f.Add(i);
    }
    std::cout << "Bloom Filter Add Timer: " << timer.getElapsedTime().count() << std::endl;

    timer.reset();
    false_pos = 0;
    for (unsigned i=2000; i<4000; ++i) {
        if (simd_b_f.Find(i)) {
            ++false_pos;
        }
    }
    std::cout << "Bloom Filter Check time: " << timer.getElapsedTime().count() << std::endl;
    std::cout << "SimdBlockFilterFixed64: " << false_pos << std::endl;

    std::vector<std::size_t> inp(2000);
    for (std::size_t i=0; i<2000; ++i) {
        inp[i] = i;
    }

    // timer.reset();
    // xor8_t filter;
    // xor8_allocate(2500, &filter);
    // xor8_populate(inp.data(), 2000, &filter);

    // std::cout << "xor8 filter Add Timer: " << timer.getElapsedTime().count() << std::endl;

    // std::cout << "xor8 memory use: " << xor8_size_in_bytes(&filter) << std::endl;

    // timer.reset();
    // false_pos = 0;
    // for (unsigned i=2000; i<4000; ++i) {
    //     if (xor8_contain(i, &filter)) {
    //         ++false_pos;
    //     }
    // }
    // std::cout << "xor8 Check timer: " << timer.getElapsedTime().count() << std::endl;
    // std::cout << "xor8 false: " << false_pos << std::endl;

    return 0;
}