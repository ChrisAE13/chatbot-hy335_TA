import os
import gradio as gr
import config
import time
from llama_index import(
    StorageContext,
    ServiceContext,
    ChatPromptTemplate,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.llms import ChatMessage, MessageRole

# get OPENAI key from system vars
# use os.environ["OPENAI_API_KEY"] = "..." to use other key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

def promt(): 
    print('\n>> ', end='') 

def user(user_message, history) -> tuple:
    return "", history + [[user_message, None]]

def generate_response(history):
    streaming_response = query_engine.query(history[-1][0])
    history[-1][1] = ""
    # print char by char
    for text in streaming_response.response_gen:
        for character in text:
            history[-1][1] += character
            # time.sleep(0.05)
            yield history

# load index from storage
storage_context = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
index = load_index_from_storage(storage_context)

# TODO a separate file that will have the prompts ??
# # Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "If the context isn't helpful, do not answer the question."
            # "Always answer the question, even if the context isn't helpful."
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Do not give me an answer if it is not mentioned in the context as a fact. \n"
            "Given this information, please provide me with an answer to the following:\n{query_str}\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# set llm model and query engine
llm = OpenAI(temperature=config.TEMPERATURE, model=config.MODEL)
service_context = ServiceContext.from_defaults(llm=llm, system_prompt="You are a teaching assistant for the course cs335b - Computer Networks")
query_engine = index.as_query_engine(service_context=service_context, streaming=True, text_qa_template=text_qa_template)


with gr.Blocks() as demo:
    chatbot = gr.components.Chatbot(label='ChatBGP', height=500)
    msg     = gr.components.Textbox(label='')
    submit  = gr.components.Button(value='Submit')
    clear   = gr.components.ClearButton()

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_response, chatbot, chatbot
    )
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_response, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()