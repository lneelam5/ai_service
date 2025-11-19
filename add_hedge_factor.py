from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import requests
import json
import os
import re
import sys
from typing_extensions import TypedDict
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

# Step 1. Define LLM (lazy initialization to avoid blocking on import)
print("Initializing ChatBedrock LLM...")
try:
    llm = ChatBedrock(
        model_id="amazon.nova-lite-v1:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0.7, "maxTokens": 512},
    )
    print("ChatBedrock LLM initialized successfully")
except Exception as e:
    print(f"Error initializing ChatBedrock: {e}", file=sys.stderr)
    raise

# Step 2. Define function (tool) to call API
def call_hedge_factor_api(sellerNumber: str, hedgeFactor: float):
    print("Calling hedge factor API...")
    payload = {
        "sellerNumber": sellerNumber,
        "hedgeFactor": hedgeFactor
    }
    print("Payload:", payload)

    # Replace with your actual API endpoint
    api_port = os.getenv("API_PORT", "8000")
    url = f"http://localhost:{api_port}/api/update-hedge-factor"
    response = requests.post(url, json=payload)
    return response.json()

# Step 3. Define the state
class AgentState(TypedDict):
    input: str
    sellerNumber: Optional[str]
    hedgeFactor: Optional[float]
    api_response: Optional[dict]

# Step 4. Define graph nodes
def parse_user_input(state: AgentState) -> AgentState:
    """Use the LLM to extract structured info from user input."""
    print("Parsing user input...")
    user_input = state.get("input", "")

    prompt = f"""Extract the seller number and hedge factor (in bps) from the text below:
"{user_input}"

Return ONLY valid JSON in this exact format (no markdown, no code blocks, no explanation):
{{"sellerNumber": "123450001", "hedgeFactor": 0.0025}}

Important:
- Convert basis points to decimal form (25bps = 0.0025, 100bps = 0.01)
- Return only the JSON object, nothing else
- sellerNumber should be a string
- hedgeFactor should be a decimal number
"""
    # ChatBedrock expects a list of messages
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    # Try to extract JSON from the response (handle markdown code blocks)
    if "```" in response_text:
        # Extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
    elif response_text.startswith("{") and response_text.endswith("}"):
        # Already JSON
        pass
    else:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"sellerNumber"[^{}]*"hedgeFactor"[^{}]*\}', response_text)
        if json_match:
            response_text = json_match.group(0)
    
    print(f"LLM response: {response_text}")
    
    try:
        structured = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response text: {response_text}")
        raise ValueError(f"Failed to parse LLM response as JSON: {response_text}")
    
    # Return updated state
    return {
        **state,
        "sellerNumber": structured.get("sellerNumber"),
        "hedgeFactor": structured.get("hedgeFactor"),
    }

def call_api_node(state: AgentState) -> AgentState:
    result = call_hedge_factor_api(state["sellerNumber"], state["hedgeFactor"])
    return {
        **state,
        "api_response": result,
    }

# Step 5. Build graph
graph = StateGraph(AgentState)
graph.add_node("parse_user_input", parse_user_input)
graph.add_node("call_api_node", call_api_node)
graph.add_edge(START, "parse_user_input")
graph.add_edge("parse_user_input", "call_api_node")
graph.add_edge("call_api_node", END)

# Step 6. Compile
agent = graph.compile()

# Step 7. Run example
# if __name__ == "__main__":
#     result = agent.invoke({
#         "input": "add a hedge factor of 25bps for seller number 123450001"
#     })
#     print("API Result:", result["api_response"])
