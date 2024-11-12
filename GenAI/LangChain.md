# LangChain

LangChain is a framework for developing applications powered by language models.


## Initiation

```py
import os
os.environ("OPENAI_API_KEY") = key
```

## Chat models

```py
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-4o-mini')
model.invoke("What is the capital of France?")
```

```py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

text = "What would be a good company name for a company that makes colorful socks?"

messages = [
    SystemMessage(content="You are a helpful assistant that generates company names"),
    HumanMessage(content=text)
]

result = model.invoke(messages)
```

## Chat Prompt Template

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates company names"),
    ("human", "{text}")
])

result = chat_prompt_template.invoke({
    "text": "What would be a good company name for a company that makes colorful socks?"
})

model = ChatOpenAI(model='gpt-4o-mini')

ai_llm_result = model.invoke(result)
```

## Output Parsing

Output parsers are classes that help structure language model responses. There are two main methods an output parser must implement:

- Get format instructions: A method which returns a string containing instructions for how the output of a language model should be formatted.
"Parse": A method which takes in a string (assumed to be the response from a language model) and parses it into some structure.
And then one optional one:

- Parse with prompt: A method which takes in a string (assumed to be the response from a language model) and a prompt (assumed to the prompt that generated such a response) and parses it into some structure. The prompt is largely provided in the event the OutputParser wants to retry or fix the output in some way, and needs information from the prompt to do so.

```py
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

model = ChatOpenAI()

class Joke(BaseModel):
    setup: str = Field(description="The setup to the joke")
    punchline: str = Field(description="The punchline to the joke")

class Jokes(BaseModel):
    jokes: List[Joke] = Field(description="A list of jokes")

parser = PydanticOutputParser(pydantic_object=Joke)

template = "Answer the user query.\n{format_instructions}\n{query}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

messages = chat_prompt.invoke({
    "query": "What is a really funny joke about Python programming?",
    "format_instructions": parser.get_format_instructions()
})

result = model.invoke(messages)

try:
    joke_object = parser.parse(result.content)
    print(joke_object.setup)
    print(joke_object.punchline)
except Exception as e:
    print(e)

structured_llm = model.with_structured_output(Joke)
result = structured_llm.invoke("What is a really funny joke about Python programming?")

print(result.punchline)
print(type(result))
```

## Summarizing

```py
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
chain = load_summarize_chain(llm, chain_type="stuff")

chain.invoke(docs)
```

```py
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, StuffDocumentsChain

llm = ChatOpenAI(temperature=0)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)

# map_chain:
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

print(map_reduce_chain.invoke(split_docs))

prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": split_docs}, return_only_outputs=True)

# Page 1 --> Page 2 (Refine) --> Page 3 (Refine)
```

## Document Loaders and Text Splitting

```py
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
import requests

# Get this file and save it locally:
url = "https://github.com/hammer-mt/thumb/blob/master/README.md"

# Save it locally:
r = requests.get(url)

# Extract the text from the HTML:
soup = BeautifulSoup(r.text, 'html.parser')
text = soup.get_text()

with open("README.md", "w") as f:
    f.write(text)

loader = TextLoader('README.md')
docs = loader.load()

from langchain_core.documents import Document
[ Document(page_content='test', metadata={'test': 'test'}) ]

# Split the text into sentences:
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)

final_docs = text_splitter.split_documents(loader.load())

from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

chat = ChatOpenAI()

chain = load_summarize_chain(llm=chat, chain_type="map_reduce")
chain.invoke({
    "input_documents": final_docs,
})
```

## Tagging Documents

```py
# fixes a bug with asyncio and jupyter
import nest_asyncio

nest_asyncio.apply()

from langchain.document_loaders.sitemap import SitemapLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
import pandas as pd

sitemap_loader = SitemapLoader(web_path="https://understandingdata.com/sitemap.xml")
sitemap_loader.requests_per_second = 5
docs = sitemap_loader.load()

# Schema
schema = {
    "properties": {
        "sentiment": {"type": "string" },
        "aggressiveness": {"type": "integer"},
        "primary_topic": {"type": "string", "description": "The main topic of the document."},
    },
     "required": ["primary_topic", "sentiment", "aggressiveness"],
}

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
chain = create_tagging_chain(schema, llm, output_key='output')

results = []

# Remove the 0:10 to run on all documents:
for index, doc in enumerate(docs[0:10]):
    print(f"Processing doc {index +1}")
    chain_result = chain.invoke({'input': doc.page_content})
    results.append(chain_result['output'])

df = pd.DataFrame(results)

print(docs[0].metadata)

# Combine the URLs with the results
df['url'] = [doc.metadata['source'] for doc in docs[0:10]]
```

## Tracing

```py
import os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"

# # Used by the agent in this tutorial
os.environ["OPENAI_API_KEY"] = ""
# os.environ["SERPAPI_API_KEY"] = "SERPAPI_API_KEY"

from langsmith import Client

client = Client()

from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

llm = ChatOpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
)

import asyncio

inputs = [
    "How many people live in canada as of 2023?",
    "who is dua lipa's boyfriend? what is his age raised to the .43 power?",
    "what is dua lipa's boyfriend age raised to the .43 power?",
    "how far is it from paris to boston in miles",
    "what was the total number of points scored in the 2023 super bowl? what is that number raised to the .23 power?",
    "what was the total number of points scored in the 2023 super bowl raised to the .23 power?",
    "how many more points were scored in the 2023 super bowl than in the 2022 super bowl?",
    "what is 153 raised to .1312 power?",
    "who is kendall jenner's boyfriend? what is his height (in inches) raised to .13 power?",
    "what is 1213 divided by 4345?",
]
results = []


async def arun(agent, input_example):
    try:
        return await agent.arun(input_example)
    except Exception as e:
        # The agent sometimes makes mistakes! These will be captured by the tracing.
        return e


for input_example in inputs:
    results.append(arun(agent, input_example))
results = await asyncio.gather(*results)

from langchain.callbacks.tracers.langchain import wait_for_all_tracers

# Logs are submitted in a background thread to avoid blocking execution.
# For the sake of this tutorial, we want to make sure
# they've been submitted before moving on. This is also
# useful for serverless deployments.
wait_for_all_tracers()
```

## Hub

```py
from langchain import hub

prompt = hub.pull("homanp/question-answer-pair")
prompt_two = hub.pull("gitmaxd/synthetic-training-data")
prompt_three = hub.pull("rlm/text-to-sql")
rag_prompt = hub.pull("rlm/rag-prompt")

# Load docs
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# Store splits
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# LLM
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
question = "What are the approaches to Task Decomposition?"
result = qa_chain.invoke({"query": question})
result["result"]
```


