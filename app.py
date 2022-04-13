import streamlit as st
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline

model_name = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
model_name = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return model

qa = load_qa_model()

st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'])
