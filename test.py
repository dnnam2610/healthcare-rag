import json 

with open('data/testData/router/test_non_med.json', 'r') as f:
    data = json.load(f)

print(len(data))