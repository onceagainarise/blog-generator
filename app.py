import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import ctransformers

def getllamaresponse(input_text, no_words, blog_style):
    config = {'max_new_tokens': 256, 'temperature': 0.01}

    # Instantiate the ctransformers object correctly
    llm = ctransformers.CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                                    model_type='llama',
                                    config=config)

    template = """
    write a blog for {blog_style} job profile for a topic {input_text} within
    {no_words} words."""

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs ")
input_text = st.text_input("enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('no of words')

with col2:
    blog_style = st.selectbox('writing the blog for', ('researchers', 'data scientist', 'common people'), index=0)

submit = st.button("generate")

if submit:
    st.write(getllamaresponse(input_text, no_words, blog_style))
