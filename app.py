import streamlit as st
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

st.title("Ask Questions about your Text")
context = st.text_area('Please paste your article :', height=30)
Q = st.text_input("Questions from this article?")
button = st.button("Get me Answers")

st.write(Q[0])
st.write(context)

with st.spinner("Discovering Answers.."):
    if button and context:
        answers = nlp({'question': Q[0],'context': context}).get('answer')
        st.write(answers['answer'])
