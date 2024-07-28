from fastapi import FastAPI
from routes import combined, query_llm, stream_request, moa_request, groq_query, llamaindex_query, friends_webook, multion_api

OPENAI_API_KEY='AIHERO GOT ME OMGGGGGGGGGGGG ALERRRRTTTT SUPER SECURE SECRETTT'

app = FastAPI()

app.include_router(query_llm.router)
app.include_router(stream_request.router)
app.include_router(moa_request.router)
app.include_router(combined.router)
app.include_router(groq_query.router)
app.include_router(llamaindex_query.router)
app.include_router(friends_webook.router)
app.include_router(multion_api.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import os
from together import AsyncTogether, Together
from dotenv import load_dotenv
import os
import json
import time
import requests
import openai
import copy

from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

load_dotenv()

router = APIRouter()

class MoaRequest(BaseModel):
    user_prompt: str

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

async def run_llm(async_client, model, user_prompt):
    """Run a single LLM call with a reference model."""
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content

@router.post("/moa_request")
async def moa_request(request: MoaRequest):
    user_prompt = request.user_prompt
    async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))

    try:
        results = await asyncio.gather(*[run_llm(async_client, model, user_prompt) for model in reference_models])

        finalStream = client.chat.completions.create(
            model=aggregator_model,
            messages=[
                {"role": "system", "content": aggregator_system_prompt},
                {"role": "user", "content": ",".join(str(element) for element in results)},
            ],
            stream=True,
        )

        content = ""
        for chunk in finalStream:
            content += chunk.choices[0].delta.content or ""

        return {"response": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import textgrad as tg
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query_llm")
async def query_llm(request: QueryRequest):
    """
    Endpoint to query the LLM with a given question and return the LLM's answer.

    Args:
        request (QueryRequest): The request object containing the question.

    Returns:
        dict: A dictionary with the answer from the LLM.

    Raises:
        HTTPException: If there is an error during the query.
    """
    # Extract the question from the request
    question_string = request.question

    try:
        # Initialize the LLM engine and model
        llm_engine = tg.get_engine("gpt-3.5-turbo")
        tg.set_backward_engine("gpt-4o", override=True)
        model = tg.BlackboxLLM("gpt-4o")

        # Create a variable for the question and set its role description
        question = tg.Variable(question_string,
                               role_description="question to the LLM",
                               requires_grad=False)

        # Get the answer from the model
        answer = model(question)

        # Set the role description for the answer
        answer.set_role_description("concise and accurate answer to the question")

        # Create an optimizer and a loss function
        optimizer = tg.TGD(parameters=[answer])
        evaluation_instruction = (f"Here's a question: {question_string}. "
                                  "Evaluate any given answer to this question, "
                                  "be smart, logical, and very critical. "
                                  "Just provide concise feedback.")
        loss_fn = tg.TextLoss(evaluation_instruction)

        # Compute the loss and perform the backward pass
        loss = loss_fn(answer)
        loss.backward()

        # Update the answer
        optimizer.step()

        # Return the answer
        return {"answer": str(answer)}

    except Exception as e:
        # Raise an HTTPException if there is an error
        raise HTTPException(status_code=500, detail=str(e))

import os
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "mixtral-8x7b-32768",
            "temperature": 0.1,
            "max_tokens": 1000,
        }
    }
}
memory_instanence = Memory.from_config(config)



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import nest_asyncio

# Ensure nest_asyncio is applied before any event loop is created
# nest_asyncio.apply()

router = APIRouter()

class QueryRequest(BaseModel):
    user_prompt: str

@router.post("/")
async def groq_query(request: QueryRequest):
    try:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in environment variables")

        # Initialize Embeddings without extra parameters
        embed_model = FastEmbedEmbeddings()

        chat_model = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

        vectorstore = Chroma(embedding_function=embed_model, persist_directory="chroma_db_llamaparse1", collection_name="rag")

        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

        custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

        qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})

        response = qa.invoke({"query": request.user_prompt})

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException
import os
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import joblib
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    llamaparse_api_key = os.environ.get("LLAMAPARSE_API_KEY")

    if not llamaparse_api_key:
        raise ValueError("LLAMAPARSE_API_KEY is not set in environment variables")

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionSOC2 = """
SOC-2 Compliance Report Parsing Instruction.
"""
        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionSOC2,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data("./data/aws-soc-2.pdf")

        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.
    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    print(llama_parse_documents[0].text[:300])

    with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "./data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",
        collection_name="rag"
    )

    print('Vector DB created successfully !')
    return vs, embed_model

@router.post("/llamaindex_query")
async def create_vector_db():
    try:
        vs, embed_model = create_vector_database()
        return {"message": "Vector DB created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from typing import List, Dict
from fastapi import APIRouter, HTTPException, FastAPI
from pydantic import BaseModel

router = APIRouter()

class WebhookRequest(BaseModel):
    url: str
    command: str

    def __str__(self):
        return f"WebhookRequest(url={self.url}, command={self.command})"

class WebhookResponse(BaseModel):
    response: Dict[str, str]

    def __str__(self):
        return f"WebhookResponse(response={self.response})"

class WebhookEntry(BaseModel):
    request: WebhookRequest
    response: WebhookResponse

    def __str__(self):
        return f"WebhookEntry(request={self.request}, response={self.response})"

# In-memory storage for webhook data
webhook_storage: str = '{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"Input should be a valid dictionary or object to extract fields from","input":"{  \n  \"id\": 1,                    \n  \"createdAt\": \"2024-07-21T12:34:56\",\n  \"transcript\": \"transcript\",\n  \"structured\": {\n    \"title\": \"title\",\n    \"overview\": \"overview\",\n    \"emoji\": \"emoji\",\n    \"category\": \"category\",\n    \"actionItems\": [\"Action item 1\", \"Action item 2\"]\n  },\n  \"pluginsResponse\": [\"This is a plugin response item github is sakomws \"],\n  \"discarded\": false\n}"}]}%'

@router.post("/webhook", response_model=WebhookEntry)
def save_webhook_data(webhook_entry: WebhookEntry):
    global webhook_storage
    webhook_storage += str(webhook_entry) + "\n"
    print('sako',webhook_storage)
    return webhook_entry

@router.get("/webhook", response_model=str)
def get_webhook_data():
    print('webhook_storage:', webhook_storage)
    return webhook_storage
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from multion.client import MultiOn
import os
from dotenv import load_dotenv
import agentops
import requests

load_dotenv()

# agentops.init(os.getenv("AGENTOPS_API_KEY"))
multion_api_key = os.getenv("MULTION_API_KEY")
multion = MultiOn(api_key=multion_api_key)

router = APIRouter()

class Memory(BaseModel):
    url: str
    command: str

@router.post("/multion_webhook")
async def webhook(memory: Memory):
    try:
        if not multion_api_key:
            raise HTTPException(status_code=500, detail="API key not found.")
        
        # Perform the browse operation using MultiOn client
        browse = multion.browse(
            cmd=memory.command,
            url=memory.url,
            include_screenshot=True
        )
        
        # Fetch data from the /webhook endpoint
        response = requests.get("http://127.0.0.1:8000/webhook")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch webhook data.")
        
        webhook_data = response
        if not webhook_data:
            raise HTTPException(status_code=404, detail="No webhook data found.")
    
        print("Browse response:", browse)
        return {
            "webhook_data": webhook_data,
            "browse_response": browse
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import os
import asyncio
from dotenv import load_dotenv
from aiohttp import ClientSession
from together import AsyncTogether, Together

from .query_llm import query_llm, QueryRequest
from .stream_request import stream_request, StreamingRequest
from .moa_request import run_llm, MoaRequest

load_dotenv()

router = APIRouter()

class CombinedRequest(BaseModel):
    question: str
    streaming_request: dict

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


@router.post("/combined")
async def combined_endpoint(request: CombinedRequest):
    question_string = request.question
    streaming_request_data = request.streaming_request

    async with ClientSession() as session:
        async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"), session=session)
        try:
            # Call query_llm endpoint
            llm_response = await query_llm(QueryRequest(question=question_string))
            llm_answer = llm_response['answer']

            # Call stream_request endpoint
            stream_response = await stream_request(StreamingRequest(**streaming_request_data))
            streaming_content = stream_response['response']

            # Run Mixture-of-Agents logic
            results = await asyncio.gather(*[run_llm(async_client, model, question_string) for model in reference_models])

            finalStream = client.chat.completions.create(
                model=aggregator_model,
                messages=[
                    {"role": "system", "content": aggregator_system_prompt},
                    {"role": "user", "content": ",".join(str(element) for element in results)},
                ],
                stream=True,
            )

            moa_content = ""
            for chunk in finalStream:
                moa_content += chunk.choices[0].delta.content or ""

            return {
                "llm_answer": llm_answer,
                "streaming_response": streaming_content,
                "moa_response": moa_content
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class StreamingRequest(BaseModel):
    inputs: List[dict]
    max_tokens: int
    stop: List[str]
    model: str

SAMBANOA_API_URL = os.getenv('SAMBANOA_API_URL')
SAMBANOA_API_KEY = os.getenv('SAMBANOA_API_KEY')

@router.post("/stream_request")
async def stream_request(request: StreamingRequest):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic " + SAMBANOA_API_KEY
    }

    response = requests.post(SAMBANOA_API_URL, headers=headers, json=request.dict(), stream=True)

    if response.status_code == 200:
        content = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                content += chunk.decode('utf-8')
        return {"response": content}
    else:
        raise HTTPException(status_code=response.status_code, detail="Request failed")
