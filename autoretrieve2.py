import dotenv
dotenv.load_dotenv()

from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "horse"})

from langchain.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
)

final_prompt = PromptTemplate.from_template("""
You are a wondrous wizard of math.

{% for example in examples %}
Human: {{ example.input }}
AI: {{ example.output }}

{% endfor %}

Human: {{ input }}
AI:""", template_format="jinja2")

# print(few_shot_prompt.invoke({"input": "What's 3+3?"}))

from langchain.llms import OpenAI
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda


chain = {
    "input": RunnablePassthrough(),
    "examples": RunnableLambda(lambda input: example_selector.select_examples({"input": input}))
} | final_prompt | OpenAI(temperature=0.0)

print(chain.invoke("What's 3+3?"))
