import os
import config
# from pathlib import Path
from llama_index.llms import OpenAI
from llama_index import (
	SimpleDirectoryReader, 
	VectorStoreIndex,
	ServiceContext,
)

# get OPENAI key from system vars
# use os.environ["OPENAI_API_KEY"] = "..." to use other key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(temperature=config.TEMPERATURE, model=config.MODEL)
service_context = ServiceContext.from_defaults(llm=llm, system_prompt="You are a teaching assistant for the course cs335b - Computer Networks")

# clear storage director
if(os.path.exists(config.STORAGE_DIR)):
	print("Clearing", config.STORAGE_DIR, "directory")
	for file in os.listdir(config.STORAGE_DIR):
		os.remove(config.STORAGE_DIR+'/'+file)
		print(file, "deleted")

# TODO: maybe find a more efficient way of reading and storing info
# perhaps diff loaders for diff file types --> see llama_hub
dir_reader = SimpleDirectoryReader(config.DATA_DIR, recursive=True)
documents = dir_reader.load_data(show_progress=True)


# TODO: check for different type of indexing instead of vector
# check extra parameters for indexing
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir=config.STORAGE_DIR) 
print("Created storage directory:", config.STORAGE_DIR)