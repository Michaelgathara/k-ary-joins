from utils import *
# Let's illustrate the concept of binary join and k-ary join using a simple example with Python data structures.
# We'll simulate joining three tables (A, B, and C) with binary and k-ary approaches.

# Sample data for tables A, B, and C.
# Each table is represented as a list of dictionaries, where each dictionary represents a row in the table.
table_A = [{'id': 1, 'value_A': 'A1'}, {'id': 2, 'value_A': 'A2'}, {'id': 3, 'value_A': 'A3'}]
table_B = [{'id': 1, 'value_B': 'B1'}, {'id': 2, 'value_B': 'B2'}]
table_C = [{'id': 1, 'value_C': 'C1'}, {'id': 3, 'value_C': 'C3'}]

# Binary Join Implementation: Joining A->B and then (A,B)->C.
def binary_join(table1, table2, key):
    # Perform a simple equi-join based on the specified key.
    join_result = []
    for row1 in table1:
        for row2 in table2:
            if row1[key] == row2[key]:
                join_result.append({**row1, **row2})  # Merge dictionaries
    return join_result

# First, join A and B.
intermediate_result = binary_join(table_A, table_B, 'id')
# Then, join the intermediate result with C.
final_result_binary = binary_join(intermediate_result, table_C, 'id')

# K-ary Join Implementation: Joining A, B, and C in a single operation.
def k_ary_join(tables, key):
    # Assuming all tables have a unique key for simplicity.
    # Creating a dictionary to map keys to combined row data.
    join_map = {}
    for table in tables:
        print(f"{colors.OKGREEN} (31) {colors.END} Table: {table}")
        for row in table:
            print(f"{colors.OKCYAN} (34) {colors.END} Row: {row}")
            row_key = row[key]
            if row_key not in join_map:
                join_map[row_key] = {}
                print(f"{colors.WARNING} (38) {colors.END} Join Map: {join_map}")
            join_map[row_key].update(row)
    # Filter out incomplete joins, i.e., rows not present in all tables.
    return [row for row in join_map.values() if all(table in row for table in ['value_A', 'value_B', 'value_C'])]

# Perform k-ary join on A, B, and C.
final_result_k_ary = k_ary_join([table_A, table_B, table_C], 'id')

print(f"Result binary\n: {final_result_binary}\nResult K-Ary:\n{final_result_k_ary}")
print(f"Binary result size: {len(final_result_binary)}\nK-ary result size: {len(final_result_k_ary)}")