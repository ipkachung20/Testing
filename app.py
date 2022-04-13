import streamlit as st
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
model = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
Q = [ "Where do I live?" ]
context = (" USA live in UK ")

st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = nlp({
        'question': Q[0],
        'context': context
        }).get('answer')
        
        st.write(answers)
