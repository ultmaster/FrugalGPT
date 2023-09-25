import dotenv
dotenv.load_dotenv()

from operator import itemgetter

from langchain.llms import OpenAI, Cohere, AI21
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.output_parsers import BooleanOutputParser

from langchain.schema.runnable import Runnable, RouterRunnable, RunnableMap, RunnablePassthrough, RunnableLambda, RunnableSequence

class Print(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        out = super().invoke(*args, **kwargs)
        print(out)
        return out


def llm_cascade(llms: list[BaseLLM]):
    good_enough_prompt = PromptTemplate.from_template("Does the response have a good quality? Answer `YES` or `NO`.\nRequest: {input}\nResponse: {output}\nAnswer:")
    good_enough = RunnableMap({
        "input": itemgetter("input"),
        "output": itemgetter("output"),
        "quality": {
            "input": RunnablePassthrough(),
            "key": itemgetter("quality"),
        } | RouterRunnable({
            "good": lambda _: "good",
            "bad": {
                "input": itemgetter("input"),
                "output": itemgetter("output")  # eject good_enough
            } | good_enough_prompt
            | OpenAI(temperature=0.)
            | BooleanOutputParser()
            | (lambda out: "good" if out else "bad")
        })
    })
    chain = {
        "input": RunnablePassthrough(),
        "output": lambda _: None,
        "quality": lambda _: "bad"
    }
    for llm in llms:
        routed_llm = {
            "input": itemgetter("input"),
            "output": {
                "input": RunnablePassthrough(),
                "key": itemgetter("quality"),
            } | RouterRunnable({
                "bad": itemgetter("input") | llm,
                "good": itemgetter("output")
            }),
            "quality": itemgetter("quality")
        } | good_enough | Print()
        chain = chain | routed_llm
    return chain


llms = [
    OpenAI(model_name="text-ada-001", temperature=0.),
    Cohere(model="command", temperature=0.),
    AI21(),
    OpenAI(model_name="text-davinci-003", temperature=0.),
]

llm_cascade(llms).invoke("What is 31+55?")
# OpenAI(model_name="claude-2").invoke("What is 31+55?")