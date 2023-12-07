import os
import config
from llama_index import(
    StorageContext,
    ServiceContext,
    load_index_from_storage,
) 
from llama_index.llms import OpenAI

# get OPENAI key from system vars
# use os.environ["OPENAI_API_KEY"] = "..." to use other key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

def promt(): 
    print('\n>> ', end='') 

# load index from storage
storage_context = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
index = load_index_from_storage(storage_context)

# set llm model and query engine
llm = OpenAI(temperature=config.TEMPERATURE, model=config.MODEL)
service_context = ServiceContext.from_defaults(llm=llm)
query_engine = index.as_query_engine(service_context=service_context)

# chatbot loop
print('\nHello! Ask me about HY335! (Press q, quit or exit to exit.)')
promt()

query = input()

while (query!= 'q') and (query != 'quit') and (query !='exit'):
    response = query_engine.query(query)
    print(f'\n{response}')
    promt()
    query = input()

print('\nBye!')