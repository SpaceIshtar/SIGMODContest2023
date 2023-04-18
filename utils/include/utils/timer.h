//
// Created by longxiang on 3/15/23.
//

#ifndef SIGMOD_CONTEST_TIMER_H
#define SIGMOD_CONTEST_TIMER_H

#include <chrono>
#include <iostream>

namespace utils {
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
           std::cout << msg << std::endl;
           std::cout << "Time cost: " << time.count() << std::endl;
        }
    };
} // namespace utils


#endif //SIGMOD_CONTEST_TIMER_H
