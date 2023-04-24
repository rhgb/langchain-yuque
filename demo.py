from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "stabilityai/stablelm-tuned-alpha-7b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={
    "temperature": 0.7,
    "max_new_tokens": 64,
})

template = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
<|USER|>{question}<|ASSISTANT|>"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))
