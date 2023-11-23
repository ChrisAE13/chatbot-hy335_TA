import os
import config
from pathlib import Path
from llama_index.llms import OpenAI
from llama_hub.file.unstructured import UnstructuredReader
from llama_index import (
	SimpleDirectoryReader, 
	VectorStoreIndex,
	ServiceContext,
)

# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=config.TEMPERATURE, model=config.MODEL)
service_context = ServiceContext.from_defaults(llm=llm)

# TODO:
dir_reader = SimpleDirectoryReader(config.DATA_DIR, file_extractor={
	".html": UnstructuredReader(),
	".txt": UnstructuredReader(),
})
documents = dir_reader.load_data()

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.storage_context.persist(persist_dir=config.STORAGE_DIR) 
