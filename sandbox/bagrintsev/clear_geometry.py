import sys
import json

if len(sys.argv) < 2:
    print("Usage: python <path_to_json_file>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = {"graph": data["graph"]}
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
print(f"File {file_path} updated successfully.")
