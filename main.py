from agent import Agent


# Run the agent on some input
agent = Agent()
result = agent.executor.invoke({'input': 'What was Tesla\'s revenue for Q2 2023'})
print(result)

result = agent.executor.invoke({'input': 'Summarize Apple\'s latest earnings call.'})
print(result)