import json

file_path = "llada_output_truthfulqa_train0_test0_blk16_gen32_train.jsonl"

# Read and modify lines
with open(file_path, "r") as f:
    lines = []
    for line in f:
        data = json.loads(line)
        if "response" in data:
            data["generation"] = data.pop("response")
        lines.append(json.dumps(data))

# Overwrite the file with updated lines
with open(file_path, "w") as f:
    for line in lines:
        f.write(line + "\n")
