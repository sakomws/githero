import json
from datetime import datetime, timezone
from invoke_types import InvocationRequest, Actor, LLMMessage
from settings import MODEL, MODEL_KEY
import os
import requests
from crewai import Agent, Task, Crew
from langchain.llms import Ollama
from langchain_groq import ChatGroq
import git_parser, policy_parser
import agentops
import http.client
import json
import load_dotenv

load_dotenv.load_dotenv()


AGENTOPS_API_KEY = os.environ.get('AGENTOPS_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
# GROQ_MODEL_NAME='llama-3.1-8b-instant'
GROQ_MODEL_NAME='mixtral-8x7b-32768'

agentops.init(AGENTOPS_API_KEY)

llm = Ollama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# llm=ChatGroq(temperature=0,
#              model_name=GROQ_MODEL_NAME,
#              api_key=GROQ_API_KEY)

def convert_crew_output(crew_output):
    # Example conversion logic
    return json.dumps(crew_output.to_dict())

def get_actor_prompt(actor: Actor):
    return (f"You are providing information about git repository security. "
            f"Your outputs need to be informational responses. "
            f"Give elaborate visual descriptions of the repository and the security recommendations. "
            f" ")

def get_system_prompt(request: InvocationRequest):
    return request.global_story + ("Githero is to scan the git repository and share security recommendations.") + get_actor_prompt(request.actor)

def invoke_ai(conn, turn_id: int, prompt_role: str, system_prompt: str, messages: list[LLMMessage]):
    MAX_TOKENS = 1000

    with conn.cursor() as cur:
        start_time = datetime.now(tz=timezone.utc)
        serialized_messages = [msg.model_dump() for msg in messages]
        repo_url = 'https://github.com/sakomws/aiproxy.git'
        clone_dir = './repository'
        output_file = 'merged.py'
        git_parser.clone_repo(repo_url, clone_dir)
        git_parser.merge_python_files(clone_dir, output_file)
        with open("merged.py", "r") as file:
            backstory_content = file.read()
        
        backstory_content=str(backstory_content) # + str(policy_parser.scrape_github_secrets_guide("https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions") +backstory_content+policy_parser.scrape_github_secrets_guide("https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions"))

        security_agent = Agent(
            role="Security Analyst",
            goal="""Check organization's security and provide analysis on the security status.""",
            backstory="""You are a security analyst with a passion for security. You are also known for your ability to analyze the security of the organization.""",
            allow_delegation=False,
            verbose=True,
            llm=llm
        )
    
        task = Task(description="""Check the security status of the organization.""",
                     agent=security_agent,
                     expected_output="Provide security recommendations and best practices for this specific codebase. Output should be in human readable format.",)

        crew = Crew(
            agents=[security_agent],
            tasks=[task],
            verbose=2
        )

        result = str(crew.kickoff())

        key_agent = Agent(
            role="Key Manager",
            goal="""Analyze the codebase and provide a list of keys and their usage.""",
            backstory=backstory_content,
            allow_delegation=False,
            verbose=True,
            llm=llm
        )
    
        task = Task(description="""Analyze the codebase and provide a list of keys and their usage.""",
            agent = key_agent,
            expected_output="Provide a list of keys and their usage in the codebase. If there are any keys that are not secure or exposed, provide recommendations on how to secure them.")
    
        crew = Crew(
            agents=[key_agent],
            tasks=[task],
            verbose=2
        )
    
        result = result+str(crew.kickoff())

        crew_output_converted = str(result)  # Convert CrewOutput to a JSON-serializable format

        text_response = result
    
        input_toofficesecuritys = 4
        output_toofficesecuritys = 10
        total_toofficesecuritys = input_toofficesecuritys + output_toofficesecuritys

        finish_time = datetime.now(tz=timezone.utc)

        cur.execute(
            "INSERT INTO ai_invocations(conversation_turn_id, prompt_role, model, model_key, prompt_messages, system_prompt, response, started_at, finished_at, input_toofficesecuritys, output_toofficesecuritys, total_toofficesecuritys) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (turn_id, prompt_role, MODEL, MODEL_KEY, json.dumps(serialized_messages), system_prompt, crew_output_converted,
             start_time.isoformat(), finish_time.isoformat(), input_toofficesecuritys, output_toofficesecuritys, total_toofficesecuritys)
        )

    return text_response

def respond_initial(conn, turn_id: int, request: InvocationRequest):
    return invoke_ai(
        conn,
        turn_id,
        "initial",
        system_prompt=get_system_prompt(request),
        messages=request.actor.messages,
    )

def get_critique_prompt(request: InvocationRequest, last_utterance: str):
    return f"""
    """

def critique(conn, turn_id: int, request: InvocationRequest, unrefined: str) -> str:
    return invoke_ai(
        conn,
        turn_id,
        "critique",
        system_prompt=get_critique_prompt(request, unrefined),
        messages=[LLMMessage(role="user", content=unrefined)]
    )

def check_whether_to_refine(critique_chat_response: str) -> bool:
    return critique_chat_response[:4] != "NONE"

def get_refiner_prompt(request: InvocationRequest, critique_response: str):
    original_message = request.actor.messages[-1].content

    refine_out = f"""
        """

    return refine_out

def refine(conn, turn_id: int, request: InvocationRequest, critique_response: str, unrefined_response: str):
    return invoke_ai(
        conn,
        turn_id,
        "refine",
        system_prompt=get_refiner_prompt(request, critique_response),
        messages=[
            LLMMessage(
                role="user",
                content=unrefined_response,
            )
        ]
    )
