#ifndef __BINARY_KERNEL__
#define __BINARY_KERNEL__

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

struct Row {
    int key;
    float value;
};


__global__ void binaryJoinKernel(const Row* table1, size_t table1Size,
                                 const Row* table2, size_t table2Size,
                                 Row* resultTable, size_t* resultSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= table1Size) return;

    for (size_t j = 0; j < table2Size; ++j) {
        if (table1[idx].key == table2[j].key) {
            int resultIdx = atomicAdd(
                reinterpret_cast<unsigned long long*>(resultSize), 1ULL);
            resultTable[resultIdx] = {table1[idx].key,
                                      table1[idx].value + table2[j].value};
            break;
        }
    }
}

#endif