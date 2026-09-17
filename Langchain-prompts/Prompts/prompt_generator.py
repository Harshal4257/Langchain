from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
please summarize the research paper title {paper_input} with following specifications:
Explanation style : {style_input}
Explanation length : {length_input}

""",
input_variables=['paper_input','style_input','length_input'],
validate_template=True
)
template.save('template.json')