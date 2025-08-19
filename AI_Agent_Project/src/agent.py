from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from .tools import sentiment_tool

def build_agent():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [sentiment_tool]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    query = "Hãy phân tích cảm xúc của văn bản này: 'This movie was amazing and emotional!'"
    response = agent.run(query)
    print("Agent response:", response)
