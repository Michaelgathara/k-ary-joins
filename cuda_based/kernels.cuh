#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "hisa.cuh"

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


// __global__ void simpleBinaryJoinKernel(GHashRelContainer* table1, GHashRelContainer* table2, SimpleRow* result, size_t* resultSize) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Search for table1's key in table2
//     SimpleRow row = searchInGHashRelContainer(table2, table1->tuples[idx].key); // Placeholder for actual search
//     if (row.key != -1) { // Assuming -1 signifies 'not found'
//         int location = atomicAdd(resultSize, 1);
//         result[location].key = row.key;
//         result[location].value = table1->data_raw[idx].value + row.value;
//     }
// }

// __global__ void simpleBinaryJoinKernel(const GHashRelContainer* table1, const GHashRelContainer* table2,
//                                        SimpleRow* result, size_t* resultSize, size_t maxResultSize) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= table1->tuple_counts) return;

//     float value = searchInGHashRelContainer(table2, table1->tuples[idx].key);
//     if (value != -1) { // Assuming -1 signifies 'not found'
//         int location = atomicAdd(resultSize, 1);
//         if (location < maxResultSize) { // Prevent out-of-bounds access
//             result[location].key = table1->tuples[idx].key;
//             result[location].value = table1->data_raw[idx].value + value;
//         }
//     }
// }