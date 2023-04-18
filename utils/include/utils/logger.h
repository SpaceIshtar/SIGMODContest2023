//
// Created by longxiang on 3/15/23.
//

#include <string>
#include <fstream>
#include <iostream>

namespace utils{
    class Logger {
        std::string file_path_;
        bool to_file_;
    public:
        Logger(const std::string &file_path, bool to_file = false):file_path_(file_path), to_file_(to_file){}

        void log(const std::string &data);

        Logger operator<<(const std::string &data);

        Logger operator<<(const int data);

        Logger operator<<(const unsigned data);

        Logger operator<<(const float data);

    };

} // namespace utils
