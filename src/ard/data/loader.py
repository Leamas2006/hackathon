import json

def load_json_dataset(file_path):
    """Load JSON file containing scientific papers or nodes."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    dataset = load_json_dataset("data/sample_subgraph.json")
    print(f"Loaded {len(dataset)} records.")
