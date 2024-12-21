# thoughtful_ai_agent
simple QA agent that answers questions based on sample QA dataset

## What does this agent do?

- The agent accepts user input and answer the question like a conversational AI Agent.
- The agent retrieves the most relevant answer from a hardcoded set of responses about Thoughtful AI.
- The agent displays the answer to the user in a user-friendly format.

## Sample data

Available here `data\sample.json`

## Setup

Setup a python env and install required libraries

`pip install -r requirements.txt`

Setup OPENAI key

`export OPENAI_API_KEY=....`

## How to run

`streamlit run agent.py <dataset file>`

eg:

`streamlit run agent.py ./data/sample.json`

