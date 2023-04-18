//
// Created by longxiang on 3/15/23.
//

#include "utils/logger.h"

namespace utils{
    void Logger::log(const std::string &data) {
        if (to_file_) {
            std::ofstream file_stream;
            file_stream.open(file_path_.c_str(), std::ios_base::app);
            file_stream << data << std::endl;
            file_stream.close();
        } else {
            std::cout << data << std::endl;
        }
    }

    Logger Logger::operator<<(const std::string &data) {
        if (to_file_) {
            std::ofstream file_stream;
            file_stream.open(file_path_.c_str(), std::ios_base::app);
            file_stream << data;
            file_stream.close();
            return *this;
        } else {
            std::cout << data;
            return *this;
        }
    }

    Logger Logger::operator<<(const int data) {
        if (to_file_) {
            std::ofstream file_stream;
            file_stream.open(file_path_.c_str(), std::ios_base::app);
            file_stream << data;
            file_stream.close();
            return *this;
        } else {
            std::cout << data;
            return *this;
        }
    }

    Logger Logger::operator<<(const unsigned data) {
        if (to_file_) {
            std::ofstream file_stream;
            file_stream.open(file_path_.c_str(), std::ios_base::app);
            file_stream << data;
            file_stream.close();
            return *this;
        } else {
            std::cout << data;
            return *this;
        }
    }

    Logger Logger::operator<<(const float data) {
        if (to_file_) {
            std::ofstream file_stream;
            file_stream.open(file_path_.c_str(), std::ios_base::app);
            file_stream << data;
            file_stream.close();
            return *this;
        } else {
            std::cout << data;
            return *this;
        }
    }

} // namespace utils