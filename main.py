from agent import Agent


# Run the agent on some input
agent = Agent()
#result = agent.executor.invoke({'input': 'What was Tesla\'s revenue for Q2 2023'})
#print(result)

result = agent.executor.invoke({'input': 'Summarize Apple\'s latest earnings call.'})
print(result)

#result = agent.executor.invoke({'input': 'What has Airbnb management said about profitability over the last few earnings calls?'})
#print(result)

#result = agent.executor.invoke({'input': 'Summarize Spotify\'s latest conference call?'})
#print(result)

#result = agent.executor.invoke({'input': 'How many new large deals did ServiceNow sign in the last quarter?'})
#print(result)

#result = agent.executor.invoke({'input': 'What are Mark Zuckerberg\'s and Satya Nadella\'s recent comments about AI?'})
#print(result)


