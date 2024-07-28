import json
from datetime import datetime, timezone
from invoke_types import InvocationRequest, Actor, LLMMessage
from settings import MODEL, MODEL_KEY
import os
import requests
from crewai import Agent, Task, Crew
from langchain.llms import Ollama
import git_parser
import agentops

AGENTOPS_API_KEY = os.environ.get('AGENTOPS_API_KEY')

agentops.init(AGENTOPS_API_KEY)

llm = Ollama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

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
        
        security_agent = Agent(
            role="Security Analyst",
            goal="""Check organization's security and provide analysis on the security status.""",
            backstory="Your backstory content here.",
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
        Examine {request.actor.name}'s last utterance: "{last_utterance}" for severe violations of these principles: Principle A: Talking about an AI assistant. {request.actor.violation} END OF PRINCIPLES.
        Focus exclusively on the last utterance and do not consider previous parts of the dialogue. 
        Identify clear and obvious violations of the preceding principles. Off-topic conversation is allowed.
        You can ONLY reference the aforementioned principles. Do not focus on anything else. 
        Provide a concise less than 100 words explanation, quoting directly from the last utterance to illustrate each violation.  
        Think step by step before listing the principles violated. Return the exact one-word phrase "NONE!" and nothing else if no principles are violated. 
        Otherwise, after your analysis, you must list the violated principles according to the following format:
        Format: QUOTE: ... CRITIQUE: ... PRINCIPLES VIOLATED: ...
        Example of this format: QUOTE: "{request.actor.name} is saying nice things." CRITIQUE: The utterance is in 3rd person perspective. PRINCIPLES VIOLATED: Principle 2: Dialogue not in the POV of {request.actor.name}.
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
        Your job is to edit informational responses for a security monitoring tool, identifying a security compromise and malware uploaded to enterprise servers. This dialogue comes from the character {request.actor.name} in response to the following prompt: {original_message} 
        Here is the story background for {request.actor.name}: {request.actor.context} {request.actor.secret} 
        Your revised informational response must be consistent with the story background and free of the following problems: {critique_response}.
        Your output revised informational response must be from {request.actor.name}'s perspective and be as identical as possible to the original user message and consistent with {request.actor.name}'s personality: {request.actor.personality}. 
        Make as few changes as possible to the original input! 
        Omit any of the following in your output: quotation marks, commentary on story consistency, mentioning principles or violations.
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
