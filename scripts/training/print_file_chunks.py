import os
import binascii
import sys

filepath = sys.argv[1]
if not os.path.exists(filepath):
    print(f"ERROR: File {filepath} not found.")
    sys.exit(1)

with open(filepath, "rb") as f:
    hex_data = binascii.hexlify(f.read()).decode("utf-8")

chunk_size = 50000
total_len = len(hex_data)
print(f"FILE_NAME: {os.path.basename(filepath)}")
print(f"TOTAL_LENGTH: {total_len}")
print(f"CHUNKS_COUNT: {(total_len + chunk_size - 1) // chunk_size}")

for i in range(0, total_len, chunk_size):
    chunk = hex_data[i:i+chunk_size]
    print(f"CHUNK_{i // chunk_size}: {chunk}")
print("===END_OF_FILE===")
