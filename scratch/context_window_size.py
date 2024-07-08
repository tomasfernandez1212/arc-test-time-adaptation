import os 
import json

"""
This is a simple script to get the context size required based on the ARC-AGI dataset.
"""

data_dir = "../ARC-AGI/data/"
json_filenames = os.listdir(os.path.join(data_dir, "training"))
data = [json.load(open(os.path.join(data_dir, "training", filename))) for filename in json_filenames]
task_context_sizes = []

for i in range(len(data)):
    task = data[i]
    task_training_samples = task["train"]
    task_test = task["test"]

    task_context_size = 0 
    for task_training_sample in task_training_samples:
        input_x_dim = len(task_training_sample["input"])
        input_y_dim = len(task_training_sample["input"][0])

        output_x_dim = len(task_training_sample["output"])
        output_y_dim = len(task_training_sample["output"][0])

        task_context_size += input_x_dim*input_y_dim + output_x_dim*output_y_dim

    for task_test_sample in task_test:
        input_x_dim = len(task_test_sample["input"])
        input_y_dim = len(task_test_sample["input"][0])

        output_x_dim = len(task_test_sample["output"])
        output_y_dim = len(task_test_sample["output"][0])

        task_context_size += input_x_dim*input_y_dim + output_x_dim*output_y_dim


    task_context_sizes.append(task_context_size)
    print(f"Task {i}: {task_context_size}")

print(f"Average Context Size: {sum(task_context_sizes)/len(task_context_sizes)}") # 1108
print(f"Max Context Size: {max(task_context_sizes)}") # 9000
print(f"Min Context Size: {min(task_context_sizes)}") # 54
threshold = 2**12
print(f"Percent Below {threshold}: {(sum([1 for x in task_context_sizes if x < threshold])/len(task_context_sizes))*100}%") 
print(f"Task {task_context_sizes.index(max(task_context_sizes))} has the largest context of {max(task_context_sizes)}") 