#pragma once

#include <string>
#include <exception>
#include <pybind11/pybind11.h>

class FUSEException: public std::exception {
    private:
        std::string message = {};

    public:
        explicit FUSEException(const char *name, const char* file, const int line, const std::string& error) {
            message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
        }

        const char *what() const noexcept override { return message.c_str(); }
};

#define ENABLE_DEBUG_OUTPUT

#ifdef ENABLE_DEBUG_OUTPUT
#define DEBUG_CODE(cmd) \
    do { \
        std::cout << "Debug Info: [" << __FILE__ << ":" << __LINE__ << "] Executing: " \
                  << #cmd << std::endl; \
        cmd; \
    } while (0)
#else
#define DEBUG_CODE(cmd) do {} while (0)
#endif

#ifndef DEBUG_FILE
#define DEBUG_FILE() \
do { \
    printf("Test Passed: %s:%d. \n", __FILE__, __LINE__); \
} while (0)
#endif

#ifndef FUSE_HOST_ASSERT
#define FUSE_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw FUSEException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

template <typename T>
void print_object(const T &a, std::string str = "") {
    std::cout << "Object [ " << str << " ]: " << a << std::endl;
}

template <typename T>
void print_object(const std::vector<T> &a, std::string str = "") {
    std::cout << "Object [ " << str << " ]: ";
    for (auto i: a) std::cout << i << " ";
    std::cout << std::endl;
}

template <>
void print_object(const std::vector<uint8_t> &a, std::string str) {
    std::cout << "Object [ " << str << " ]: ";
    for (auto i: a) std::cout << static_cast<int>(i) << " ";
    std::cout << std::endl;
}

template <>
void print_object(const pybind11::bytearray& ba, std::string str) {
    std::string elems(ba);  // 自动转换 + 拷贝
    std::cout << "Bytearray (hex): [ " << str << "] ";
    for (unsigned char c : elems) {
        printf("%02x ", c);
    }
    std::cout << std::endl;
}
