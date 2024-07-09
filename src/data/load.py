from src.data.schema import Task 
import json

def load_and_validate_data(file_path: str) -> Task:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return Task.model_validate(data)