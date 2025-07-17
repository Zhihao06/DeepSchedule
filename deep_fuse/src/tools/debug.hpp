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

// DEBUG code
#define ENABLE_DEBUG_OUTPUT
#ifdef ENABLE_DEBUG_OUTPUT
#define DEBUG_CODE(cmd) \
    do { \
        std::ostringstream oss; \
        oss << "[Rank: " << std::getenv("RANK") << " ]" << "Debug Info: [" << __FILE__ << ":" << __LINE__ << "] Executing: " \
            << #cmd << std::endl; \
        std::cout << oss.str() << std::endl; \
        cmd; \
    } while (0)
#else
#define DEBUG_CODE(cmd) do {} while (0)
#endif

// DEBUG variables
#define DEBUG_VAR(...) \
    do { \
        std::ostringstream oss; \
        oss << "[Rank: " << std::getenv("RANK") << " ]: "; \
        log_impl(oss, __VA_ARGS__); \
        std::cout << oss.str() << std::endl; \
    } while (0)

// DEBUG timer
#ifndef DEBUG_TIMER
#define DEBUG_TIMER() \
    do { \
        auto now = std::chrono::system_clock::now(); \
        auto now_time_t = std::chrono::system_clock::to_time_t(now); \
        auto us = std::chrono::duration_cast<std::chrono::microseconds>( \
                    now.time_since_epoch()).count() % 1000000; \
        std::tm tm; \
        ::gmtime_r(&now_time_t, &tm); \
        std::ostringstream oss; \
        oss << "[Rank: " << std::getenv("RANK") << " ]: " << "Timer: [" << __FILE__ << ":" << __LINE__ << "] " \
            << "Now: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S"); \
        oss << "." << std::setw(6) << std::setfill('0') << us; \
        std::cout << oss.str() << std::endl; \
    } while (0)
#endif

template<typename T>
void log_impl(std::ostringstream& oss, const T& value) {
    oss << value;
}

template<typename T, typename... Args>
void log_impl(std::ostringstream& oss, const T& value, const Args&... args) {
    oss << value;
    log_impl(oss, args...);
}

// DEBUG file
#ifndef DEBUG_FILE
#define DEBUG_FILE() \
    do { \
        printf("[Rank: %s] Test Passed: %s:%d. \n", std::getenv("RANK"), __FILE__, __LINE__); \
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
