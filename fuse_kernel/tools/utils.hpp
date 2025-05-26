#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include "debug.hpp"

void getCudaProperty() {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    std::cout << "GPU Name: " << device_prop.name << std::endl;
    std::cout << "Number of SMs: " << device_prop.multiProcessorCount << std::endl;
}

void checkCudaMemoryUsage(const std::string& tag = "") {
    int device_id;
    cudaGetDevice(&device_id);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    std::cout << "[GPU Memory Probe - " << tag << "]" << std::endl;
    std::cout << "  Total memory:     " << totalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free memory:      " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Used memory:      " << (totalMem - freeMem) / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;
}

template <typename T>
torch::Tensor to_tensor(const T& value, const torch::TensorOptions& options) {
    std::vector<uint8_t> buffer(reinterpret_cast<const uint8_t*>(value.data()),
                                reinterpret_cast<const uint8_t*>(value.data()) + value.size());
    return torch::from_blob(buffer.data(), {static_cast<int64_t>(buffer.size())}, [buffer](void*) { /* Deleter */ }, options).clone().to(torch::kCUDA);
}

template <typename T>
T from_tensor(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
    std::vector<uint8_t> buffer(cpu_tensor.data_ptr<uint8_t>(), cpu_tensor.data_ptr<uint8_t>() + cpu_tensor.numel());
    return T(buffer.begin(), buffer.end());
}

template <>
torch::Tensor to_tensor<int32_t>(const int32_t& value, const c10::TensorOptions& options) {
    std::vector<uint8_t> buffer(
        reinterpret_cast<const uint8_t*>(&value),
        reinterpret_cast<const uint8_t*>(&value) + sizeof(value)
    );
    return torch::from_blob(buffer.data(), {static_cast<int64_t>(buffer.size())}, options).clone().to(torch::kCUDA);
}

template <>
int32_t from_tensor<int32_t>(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
    TORCH_CHECK(cpu_tensor.numel() == sizeof(int32_t), "Tensor size mismatch for int32_t");
    int32_t value;
    std::memcpy(&value, cpu_tensor.data_ptr<uint8_t>(), sizeof(value));
    return value;
}

template <>
torch::Tensor to_tensor<int64_t>(const int64_t& value, const c10::TensorOptions& options) {
    std::vector<uint8_t> buffer(
        reinterpret_cast<const uint8_t*>(&value),
        reinterpret_cast<const uint8_t*>(&value) + sizeof(value)
    );
    return torch::from_blob(buffer.data(), {static_cast<int64_t>(buffer.size())}, options).clone().to(torch::kCUDA);
}

template <>
int64_t from_tensor<int64_t>(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
    TORCH_CHECK(cpu_tensor.numel() == sizeof(int64_t), "Tensor size mismatch for int64_t");
    int64_t value;
    std::memcpy(&value, cpu_tensor.data_ptr<uint8_t>(), sizeof(value));
    return value;
}

template <>
torch::Tensor to_tensor<pybind11::bytearray>(const pybind11::bytearray& value,
                                             const torch::TensorOptions& options) {
    const char* data = nullptr;
    ssize_t size = 0;

    std::string str(value);
    data = str.data();
    size = static_cast<ssize_t>(str.size());

    return torch::from_blob(
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(data)),
        size,
        [value](void*) {},
        options
    ).clone().to(torch::kCUDA);
}

template <>
pybind11::bytearray from_tensor<pybind11::bytearray>(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
    auto data = cpu_tensor.data_ptr<uint8_t>();
    auto size = cpu_tensor.numel();
    return pybind11::bytearray(reinterpret_cast<const char*>(data), size);
}

template <>
torch::Tensor to_tensor<std::optional<pybind11::bytearray>>(const std::optional<pybind11::bytearray>& value,
                                             const torch::TensorOptions& options) {
    const char* data = nullptr;
    ssize_t size = 0;

    if (!value.has_value()) {
        return torch::empty({0}, options);
    }
    const auto& ba = *value;
    std::string str(ba);
    data = str.data();
    size = static_cast<ssize_t>(str.size());

    return torch::from_blob(
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(data)),
        size,
        [ba](void*) {},
        options
    ).clone().to(torch::kCUDA);
}

template <>
std::optional<pybind11::bytearray> from_tensor<std::optional<pybind11::bytearray>>(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor = tensor.is_cuda() ? tensor.cpu() : tensor;
    if (!cpu_tensor.defined() || cpu_tensor.numel() == 0) {
        return std::nullopt;
    }

    auto* data = cpu_tensor.data_ptr<uint8_t>();
    auto size = cpu_tensor.numel();

    return pybind11::bytearray(reinterpret_cast<const char*>(data), size);
}

torch::Tensor expand_tensor_to_size(const torch::Tensor& input, const std::vector<int64_t>& output_shape) {
    TORCH_CHECK(input.dim() == (int64_t)output_shape.size(),
                "Input and output must have the same number of dimensions");

    torch::Tensor output = torch::zeros(output_shape, input.options());

    std::vector<torch::indexing::TensorIndex> indices;
    for (int64_t d = 0; d < input.dim(); ++d) {
        indices.push_back(torch::indexing::Slice(0, input.size(d)));
    }

    output.index_put_(indices, input);
    return output;
}

template <typename T>
void all_gather_wrap(std::vector<T>& outputs, T& input, std::shared_ptr<c10d::ProcessGroupNCCL>& group) {
    int rank = group->getRank();
    int group_size = group->getSize();

    auto cpu_options = torch::TensorOptions().dtype(torch::kUInt8);
    auto gpu_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto input_tensor = to_tensor(input, cpu_options);

    std::vector<torch::Tensor> output_tensors(group_size);
    for (int i = 0; i < group_size; ++i) {
        output_tensors[i] = torch::empty(input_tensor.numel(), gpu_options);
    }

    std::vector<std::vector<torch::Tensor>> output_tensors_arr;
    output_tensors_arr.resize(1);
    output_tensors_arr[0] = output_tensors;
    std::vector<torch::Tensor> input_tensor_arr = {input_tensor};

    group->allgather(output_tensors_arr, input_tensor_arr)->wait();

    outputs.resize(group_size);
    for (int i = 0; i < group_size; ++i) {
        outputs[i] = from_tensor<T>(output_tensors[i]);
    }
}

template <typename T>
void all_gather_wrap(std::vector<std::vector<T>>& outputs, std::vector<T>& input, std::shared_ptr<c10d::ProcessGroupNCCL>& group) {
    int rank = group->getRank();
    int group_size = group->getSize();

    auto cpu_options = torch::TensorOptions().dtype(torch::kUInt8);
    auto gpu_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto input_tensor = to_tensor(input, cpu_options);

    // allgather size
    int64_t local_size = input_tensor.numel();
    std::vector<int64_t> local_sizes(group_size);
    all_gather_wrap(local_sizes, local_size, group);
    auto max_local_size = std::max_element(local_sizes.begin(), local_sizes.end());
    input_tensor = expand_tensor_to_size(input_tensor, {*max_local_size});

    // allgather vector
    std::vector<torch::Tensor> output_tensors(group_size);
    for (int i = 0; i < group_size; ++i) {
        output_tensors[i] = torch::empty(*max_local_size, gpu_options);
    }
    std::vector<std::vector<torch::Tensor>> output_tensors_arr;
    output_tensors_arr.resize(1);
    output_tensors_arr[0] = output_tensors;
    std::vector<torch::Tensor> input_tensor_arr = {input_tensor};

    group->allgather(output_tensors_arr, input_tensor_arr)->wait();

    outputs.resize(group_size);
    for (int i = 0; i < group_size; ++i) {
        outputs[i] = from_tensor<std::vector<T>>(output_tensors[i]);
    }
}
