import streamlit as st
from transformers import BertForQuestionAnswering, AutoTokenizer

model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

Q = [ "Where do I live?" ]
context = (" USA live in UK ")

nlp({
    'question': Q[0],
    'context': context
})
