import dotenv

from langchain.llms import AI21

dotenv.load_dotenv()

llm = AI21()
print(llm("Hello, world!"))