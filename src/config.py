import json

with open('../config.json') as f:
    config = json.load(f)

MODEL       = config['llm']['model']
TEMPERATURE = config['llm']['temperature']
DATA_DIR    = "../"+config['data']
STORAGE_DIR = "../"+config['storage']