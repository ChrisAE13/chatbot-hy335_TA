import os
import config
from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
from llama_index import(
    SimpleDirectoryReader, 
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
) 
from llama_index.llms import OpenAI

# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "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"

def promt(): 
    print('\n>> ', end='') 

storage_context = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
index = load_index_from_storage(storage_context)


llm = OpenAI(temperature=config.TEMPERATURE, model=config.MODEL)
service_context = ServiceContext.from_defaults(llm=llm)
query_engine = index.as_query_engine(service_context=service_context)



print('\nHello! Ask me about HY335! (Press q, quit or exit to exit.)')
promt()

query = input()

while (query!= 'q') and (query != 'quit') and (query !='exit'):
    response = query_engine.query(query)
    print(f'\n{response}')
    promt()
    query = input()

print('\nBye!')