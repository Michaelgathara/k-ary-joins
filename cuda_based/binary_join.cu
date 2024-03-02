#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

void binaryJoin(const Row* table1, size_t table1Size, const Row* table2,
                       size_t table2Size) {
    Row* d_table1;
    Row* d_table2;
    Row* d_resultTable;
    size_t* d_resultSize;

    cudaMalloc(&d_table1, table1Size * sizeof(Row));
    cudaMalloc(&d_table2, table2Size * sizeof(Row));
    cudaMalloc(&d_resultTable, (table1Size + table2Size) * sizeof(Row));
    cudaMalloc(&d_resultSize, sizeof(size_t));

    size_t initialResultSize = 0;
    cudaMemcpy(d_resultSize, &initialResultSize, sizeof(size_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_table1, table1, table1Size * sizeof(Row),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_table2, table2, table2Size * sizeof(Row),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (table1Size + blockSize - 1) / blockSize;

    binaryJoinKernel<<<numBlocks, blockSize>>>(d_table1, table1Size, d_table2,
                                               table2Size, d_resultTable,
                                               d_resultSize);

    size_t resultSize;
    cudaMemcpy(&resultSize, d_resultSize, sizeof(size_t),
               cudaMemcpyDeviceToHost);

    Row* resultTable = new Row[resultSize];
    cudaMemcpy(resultTable, d_resultTable, resultSize * sizeof(Row),
               cudaMemcpyDeviceToHost);

    cudaFree(d_table1);
    cudaFree(d_table2);
    cudaFree(d_resultTable);
    cudaFree(d_resultSize);

    std::cout << "Number of joined rows: " << resultSize << std::endl;
    for (size_t i = 0; i < resultSize; ++i) {
        std::cout << resultTable[i].key << " " << resultTable[i].value
                  << std::endl;
    }
    delete[] resultTable;
}

int main() {
    Row table1[] = {{1, 1.5f},   {2, 2.5f},   {3, 3.5f},   {4, 4.5f},
                    {5, 5.5f},   {6, 6.5f},   {7, 7.5f},   {8, 8.5f},
                    {9, 9.5f},   {10, 10.5f}, {11, 11.5f}, {12, 12.5f},
                    {13, 13.5f}, {14, 14.5f}, {15, 15.5f}, {16, 16.5f},
                    {17, 17.5f}, {18, 18.5f}, {19, 19.5f}, {20, 20.5f}};

    Row table2[] = {{5, 0.5f},  {6, 1.0f},  {7, 1.5f},  {8, 2.0f},
                    {9, 2.5f},  {10, 3.0f}, {11, 3.5f}, {12, 4.0f},
                    {13, 4.5f}, {14, 5.0f}, {15, 5.5f}, {16, 6.0f},
                    {17, 6.5f}, {18, 7.0f}, {19, 7.5f}, {20, 8.0f},
                    {21, 8.5f}, {22, 9.0f}, {23, 9.5f}, {24, 10.0f}};

    binaryJoin(table1, sizeof(table1) / sizeof(Row), table2,
                      sizeof(table2) / sizeof(Row));

    return 0;
}
