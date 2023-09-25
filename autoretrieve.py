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
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

print(few_shot_prompt.invoke({"input": "What's 3+3?"}))

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import Runnable


chain = final_prompt | ChatOpenAI(temperature=0.0)

from typing import Tuple, TypedDict

Coordinate = Tuple[str, ...]

class ExampleSelectorKMutator:

    class MutationCoordinates(TypedDict):
        example_selector: Coordinate
        llm: Coordinate

    def locate(self, application: Runnable) -> MutationCoordinates:
        ...


print(chain.invoke({"input": "What's 3+3?"}))
