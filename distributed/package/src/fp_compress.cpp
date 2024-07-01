#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>

#include <vector>
#include <stdexcept>

class CuSZpWrapper {
private:
    static const size_t GPU_THRESHOLD = 1000000; // Adjust this based on your needs

    static void checkCudaErrors(cudaError_t err) {
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

public:
    static std::tuple<torch::Tensor, size_t> compress(torch::Tensor input, float errorBound, const std::string& errorMode) {
        if (!input.is_contiguous()) {
            input = input.contiguous();
        }
        
        if (input.dtype() != torch::kFloat32) {
            throw std::runtime_error("Input tensor must be of type float32");
        }

        float* data = input.data_ptr<float>();
        size_t nbEle = input.numel();

        if (errorMode != "ABS" && errorMode != "REL") {
            throw std::runtime_error("Invalid errorMode. Must be 'ABS' or 'REL'.");
        }

        // Adjust error bound for relative mode
        if (errorMode == "REL") {
            float max_val = input.max().item<float>();
            float min_val = input.min().item<float>();
            errorBound *= (max_val - min_val);
        }

        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
        torch::Tensor cmpBytes = torch::empty(nbEle * sizeof(float), options);
        size_t cmpSize = 0;

        if (!input.is_cuda() || nbEle < GPU_THRESHOLD) {
            // CPU processing
            SZp_compress_hostptr_f32(data, cmpBytes.data_ptr<unsigned char>(), nbEle, &cmpSize, errorBound);
        } else {
            // GPU processing
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
            SZp_compress_deviceptr_f32(data, cmpBytes.data_ptr<unsigned char>(), nbEle, &cmpSize, errorBound, stream);
        }

        return std::make_tuple(cmpBytes.slice(0, cmpSize), cmpSize);
    }

    static torch::Tensor decompress(torch::Tensor compressed, size_t nbEle, size_t cmpSize, float errorBound) {
        if (!compressed.is_contiguous()) {
            compressed = compressed.contiguous();
        }
        
        if (compressed.dtype() != torch::kUInt8) {
            throw std::runtime_error("Compressed tensor must be of type uint8");
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(compressed.device());
        torch::Tensor result = torch::empty(nbEle, options);

        if (!compressed.is_cuda() || nbEle < GPU_THRESHOLD) {
            // CPU processing
            SZp_decompress_hostptr_f32(result.data_ptr<float>(), compressed.data_ptr<unsigned char>(), nbEle, cmpSize, errorBound);
        } else {
            // GPU processing
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
            SZp_decompress_deviceptr_f32(result.data_ptr<float>(), compressed.data_ptr<unsigned char>(), nbEle, cmpSize, errorBound, stream);
        }

        return result;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress", &CuSZpWrapper::compress, "CuSZp compression");
    m.def("decompress", &CuSZpWrapper::decompress, "CuSZp decompression");
}
