#include <iostream>
#include "kernels.cuh"
#include "hisa.cuh"
#include "tuple.cuh"


struct Row {
    int key;
    float value;
};

// Example function to insert simple row data into GHashRelContainer
void insertRow(GHashRelContainer* container, const Row& row) {
    // Pseudo-code - actual implementation depends on GHashRelContainer specifics
    container->insert(row.key, row.value);
}

// Function to initialize GHashRelContainer from an array of Row
void initializeGHashRelContainerFromRows(GHashRelContainer* container, const Row* rows, size_t numRows) {
    // Allocate or prepare 'container' as needed
    for (size_t i = 0; i < numRows; ++i) {
        insertRow(container, rows[i]);
    }
}

__device__ float searchInGHashRelContainer(GHashRelContainer* table, int key) {
    // Simplified example - would actually involve hash table lookup
    return table->search(key);  // Assuming this returns a float value directly, or -1 if not found
}

void initializeGHashRelContainerFromRows(GHashRelContainer* container, const Row* rows, size_t numRows);
void freeGHashRelContainer(GHashRelContainer* container);

GHashRelContainer table1, table2; 
Row table1Data[] = {{1, 1.0f}, {2, 2.0f}, {3, 3.0f}};
Row table2Data[] = {{2, 5.0f}, {3, 6.0f}, {4, 7.0f}};
size_t numTable1Rows = sizeof(table1Data) / sizeof(Row);
size_t numTable2Rows = sizeof(table2Data) / sizeof(Row);

initializeGHashRelContainerFromRows(&table1, table1Data, numTable1Rows);
initializeGHashRelContainerFromRows(&table2, table2Data, numTable2Rows);

Row* resultRows;
size_t resultSize = 0;


dim3 blockSize(256); // For example
dim3 gridSize((numTable1Rows + blockSize.x - 1) / blockSize.x);

simpleBinaryJoinKernel<<<gridSize, blockSize>>>(/* arguments */);

// Copy results back to host from `resultRows` and `resultSize`
// Process or display the join results


freeGHashRelContainer(&table1);
freeGHashRelContainer(&table2);