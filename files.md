I have a LiteLLM proxy running at localhost:4001 exposing the model openrouter/meta-llama/llama-3-8b:free.

I want to add this as a new agent to the agno-agi/agent-api repository with minimal changes. 

Here are the key repository files:

<files>
  <file name="README.md">
Simple Agent API
Welcome to the Simple Agent API: a robust, production-ready application for serving Agents as an API. It includes:

A FastAPI server for handling API requests.
A PostgreSQL database for storing Agent sessions, knowledge, and memories.
A set of pre-built Agents to use as a starting point.
For more information, checkout Agno and give it a ⭐️

Quickstart
Follow these steps to get your Agent API up and running:

Prerequisites: docker desktop should be installed and running.

Clone the repo
git clone https://github.com/agno-agi/agent-api.git
cd agent-api
Configure API keys
We use GPT 4.1 as the default model, please export the OPENAI_API_KEY environment variable to get started.

export OPENAI_API_KEY="YOUR_API_KEY_HERE"
Note: You can use any model provider, just update the agents in the /agents folder.

Start the application
Run the application using docker compose:

docker compose up -d
This command starts:

The FastAPI server, running on http://localhost:8000.
The PostgreSQL database, accessible on localhost:5432.
Once started, you can:

Test the API at http://localhost:8000/docs.
Connect to Agno Playground or Agent UI
Open the Agno Playground.
Add http://localhost:8000 as a new endpoint. You can name it Agent API (or any name you prefer).
Select your newly added endpoint and start chatting with your Agents.
https://github.com/user-attachments/assets/a0078ade-9fb7-4a03-a124-d5abcca6b562

Stop the application
When you're done, stop the application using:

docker compose down
Prebuilt Agents
The /agents folder contains pre-built agents that you can use as a starting point.

Web Search Agent: A simple agent that can search the web.
Agno Assist: An Agent that can help answer questions about Agno.
Important: Make sure to load the agno_assist knowledge base before using this agent.
Finance Agent: An agent that uses the YFinance API to get stock prices and financial data.
Development Setup
To setup your local virtual environment:

Install uv
We use uv for python environment and package management. Install it by following the the uv documentation or use the command below for unix-like systems:

curl -LsSf https://astral.sh/uv/install.sh | sh
Create Virtual Environment & Install Dependencies
Run the dev_setup.sh script. This will create a virtual environment and install project dependencies:

./scripts/dev_setup.sh
Activate Virtual Environment
Activate the created virtual environment:

source .venv/bin/activate
(On Windows, the command might differ, e.g., .venv\Scripts\activate)

Managing Python Dependencies
If you need to add or update python dependencies:

Modify pyproject.toml
Add or update your desired Python package dependencies in the [dependencies] section of the pyproject.toml file.

Generate requirements.txt
The requirements.txt file is used to build the application image. After modifying pyproject.toml, regenerate requirements.txt using:

./scripts/generate_requirements.sh
To upgrade all existing dependencies to their latest compatible versions, run:

./scripts/generate_requirements.sh upgrade
Rebuild Docker Images
Rebuild your Docker images to include the updated dependencies:

docker compose up -d --build
Community & Support
Need help, have a question, or want to connect with the community?

📚 Read the Agno Docs for more in-depth information.
💬 Chat with us on Discord for live discussions.
❓ Ask a question on Discourse for community support.
🐛 Report an Issue on GitHub if you find a bug or have a feature request.
Running in Production
This repository includes a Dockerfile for building a production-ready container image of the application.

The general process to run in production is:

Update the scripts/build_image.sh file and set your IMAGE_NAME and IMAGE_TAG variables.
Build and push the image to your container registry:
./scripts/build_image.sh
Run in your cloud provider of choice.
Detailed Steps
Configure for Production
Ensure your production environment variables (e.g., OPENAI_API_KEY, database connection strings) are securely managed. Most cloud providers offer a way to set these as environment variables for your deployed service.
Review the agent configurations in the /agents directory and ensure they are set up for your production needs (e.g., correct model versions, any production-specific settings).
Build Your Production Docker Image
Update the scripts/build_image.sh script to set your desired IMAGE_NAME and IMAGE_TAG (e.g., your-repo/agent-api:v1.0.0).

Run the script to build and push the image:

./scripts/build_image.sh
Deploy to a Cloud Service With your image in a registry, you can deploy it to various cloud services that support containerized applications. Some common options include:
Serverless Container Platforms:

Google Cloud Run: A fully managed platform that automatically scales your stateless containers. Ideal for HTTP-driven applications.
AWS App Runner: Similar to Cloud Run, AWS App Runner makes it easy to deploy containerized web applications and APIs at scale.
Azure Container Apps: Build and deploy modern apps and microservices using serverless containers.
Container Orchestration Services:

Amazon Elastic Container Service (ECS): A highly scalable, high-performance container orchestration service that supports Docker containers. Often used with AWS Fargate for serverless compute or EC2 instances for more control.
Google Kubernetes Engine (GKE): A managed Kubernetes service for deploying, managing, and scaling containerized applications using Google infrastructure.
Azure Kubernetes Service (AKS): A managed Kubernetes service for deploying and managing containerized applications in Azure.
Platform as a Service (PaaS) with Docker Support

Railway.app: Offers a simple way to deploy applications from a Dockerfile. It handles infrastructure, scaling, and networking.
Render: Another platform that simplifies deploying Docker containers, databases, and static sites.
Heroku: While traditionally known for buildpacks, Heroku also supports deploying Docker containers.
Specialized Platforms:

Modal: A platform designed for running Python code (including web servers like FastAPI) in the cloud, often with a focus on batch jobs, scheduled functions, and model inference, but can also serve web endpoints.
The specific deployment steps will vary depending on the chosen provider. Generally, you'll point the service to your container image in the registry and configure aspects like port mapping (the application runs on port 8000 by default inside the container), environment variables, scaling parameters, and any necessary database connections.

Database Configuration
The default docker-compose.yml sets up a PostgreSQL database for local development. In production, you will typically use a managed database service provided by your cloud provider (e.g., AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL) for better reliability, scalability, and manageability.
Ensure your deployed application is configured with the correct database connection URL for your production database instance. This is usually set via an environment variables.
  </file>
  <file name="agents/web_agent.py">
from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools

from db.session import db_url


def get_web_agent(
    model_id: str = "gpt-4.1",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Web Search Agent",
        agent_id="web_search_agent",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id=model_id),
        # Tools available to the agent
        tools=[DuckDuckGoTools()],
        # Description of the agent
        description=dedent("""\
            You are WebX, an advanced Web Search Agent designed to deliver accurate, context-rich information from the web.

            Your responses should be clear, concise, and supported by citations from the web.
        """),
        # Instructions for the agent
        instructions=dedent("""\
            As WebX, your goal is to provide users with accurate, context-rich information from the web. Follow these steps meticulously:

            1. Understand and Search:
            - Carefully analyze the user's query to identify 1-3 *precise* search terms.
            - Use the `duckduckgo_search` tool to gather relevant information. Prioritize reputable and recent sources.
            - Cross-reference information from multiple sources to ensure accuracy.
            - If initial searches are insufficient or yield conflicting information, refine your search terms or acknowledge the limitations/conflicts in your response.

            2. Leverage Memory & Context:
            - You have access to the last 3 messages. Use the `get_chat_history` tool if more conversational history is needed.
            - Integrate previous interactions and user preferences to maintain continuity.
            - Keep track of user preferences and prior clarifications.

            3. Construct Your Response:
            - **Start** with a direct and succinct answer that immediately addresses the user's core question.
            - **Then, if the query warrants it** (e.g., not for simple factual questions like "What is the weather in Tokyo?" or "What is the capital of France?"), **expand** your answer by:
                - Providing clear explanations, relevant context, and definitions.
                - Including supporting evidence such as statistics, real-world examples, and data points.
                - Addressing common misconceptions or providing alternative viewpoints if appropriate.
            - Structure your response for both quick understanding and deeper exploration.
            - Avoid speculation and hedging language (e.g., "it might be," "based on my limited knowledge").
            - **Citations are mandatory.** Support all factual claims with clear citations from your search results.

            4. Enhance Engagement:
            - After delivering your answer, propose relevant follow-up questions or related topics the user might find interesting to explore further.

            5. Final Quality & Presentation Review:
            - Before sending, critically review your response for clarity, accuracy, completeness, depth, and overall engagement.
            - Ensure your answer is well-organized, easy to read, and aligns with your role as an expert web search agent.

            6. Handle Uncertainties Gracefully:
            - If you cannot find definitive information, if data is inconclusive, or if sources significantly conflict, clearly state these limitations.
            - Encourage the user to ask further questions if they need more clarification or if you can assist in a different way.

            Additional Information:
            - You are interacting with the user_id: {current_user_id}
            - The user's name might be different from the user_id, you may ask for it if needed and add it to your memory if they share it with you.\
        """),
        # This makes `current_user_id` available in the instructions
        add_state_in_messages=True,
        # -*- Storage -*-
        # Storage chat history and session state in a Postgres table
        storage=PostgresAgentStorage(table_name="web_search_agent_sessions", db_url=db_url),
        # -*- History -*-
        # Send the last 3 messages from the chat history
        add_history_to_messages=True,
        num_history_runs=3,
        # Add a tool to read the chat history if needed
        read_chat_history=True,
        # -*- Memory -*-
        # Enable agentic memory where the Agent can personalize responses to the user
        memory=Memory(
            model=OpenAIChat(id=model_id),
            db=PostgresMemoryDb(table_name="user_memories", db_url=db_url),
            delete_memories=True,
            clear_memories=True,
        ),
        enable_agentic_memory=True,
        # -*- Other settings -*-
        # Format responses using markdown
        markdown=True,
        # Add the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Show debug logs
        debug_mode=debug_mode,
    )
  </file>
  <file name="agents/selector.py">
from enum import Enum
from typing import List, Optional

from agents.agno_assist import get_agno_assist
from agents.finance_agent import get_finance_agent
from agents.web_agent import get_web_agent


class AgentType(Enum):
    WEB_AGENT = "web_agent"
    AGNO_ASSIST = "agno_assist"
    FINANCE_AGENT = "finance_agent"


def get_available_agents() -> List[str]:
    """Returns a list of all available agent IDs."""
    return [agent.value for agent in AgentType]


def get_agent(
    model_id: str = "gpt-4.1",
    agent_id: Optional[AgentType] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
):
    if agent_id == AgentType.WEB_AGENT:
        return get_web_agent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    elif agent_id == AgentType.AGNO_ASSIST:
        return get_agno_assist(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    elif agent_id == AgentType.FINANCE_AGENT:
        return get_finance_agent(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)

    raise ValueError(f"Agent: {agent_id} not found")
  </file>
</files>

Please provide:
1. A new agent file for the LiteLLM proxy agent
2. Updates to agents/selector.py to include the new agent
3. Updates to README.md documenting the new agent and LiteLLM configuration

Make minimal changes that follow the existing patterns in the codebase.
