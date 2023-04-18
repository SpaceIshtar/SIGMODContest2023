//
// Created by longxiang on 3/15/23.
//

#include "utils/logger.h"
#include "gtest/gtest.h"
#include <iostream>

namespace {
    TEST(LoggerTest, FileLoggerTest) {
        utils::Logger logger_tofile("/home/longxiang/log_test", true);
        logger_tofile.log("test");
        logger_tofile.log("second");
        logger_tofile << "third";
        logger_tofile << "  xxxx";
        logger_tofile << " xxx " << "yyyy" << "\n";
        logger_tofile << "4th xxx" << "\n";

        utils::Logger *logger_ptr = new utils::Logger("/home/longxiang/log_ptr", true);

        logger_ptr->log("test");
        logger_ptr->log("second");
        (*logger_ptr) << "third";
        (*logger_ptr) << "  xxxx";
        (*logger_ptr) << " xxx " << "yyyy" << "\n";
        (*logger_ptr) << "4th xxx" << "\n";

    }

}