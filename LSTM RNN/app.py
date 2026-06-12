import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Embedding,LSTM,GRU,Dropout,Dense
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('next_word_lstm.h5',compile=False)

with open('tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)

def predict_next_word(model,tokenizer,text,max_sequence_length):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_length:
        token_list=token_list[-(max_sequence_length-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_length-1,padding='pre')
    
    predicted=model.predict(token_list,verbose=0)
    
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None    
    
## Streamlit app
st.title('Next Word Guesser')
st.write("Give Your Incomplete text:")

user_input=st.text_area('Text Area:')
if st.button('Guess'):
    next_word = predict_next_word(
        model,
        tokenizer,
        user_input,
        model.input_shape[1] + 1
    )
    st.success(f"Predicted Next Word: {next_word}")