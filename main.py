# Usage: streamlit run main.py

import streamlit as st
from transformers import pipeline
import pandas as pd

clf = pipeline('text-classification', model='Helwyn/Fanfics_classification', return_all_scores=True)

text = st.text_area('Enter FanficText', height=500)

if len(text)>0:

    pred = clf(text, truncation = True)

    # st.dataframe(pred)

    labels_dict = {
        'LABEL_0': 'Explicit',
        'LABEL_1': 'Gen. audience'
    }

    scores_df = pd.DataFrame(pred[0]).sort_values(by='score')

    scores_df['label'] = scores_df['label'].apply(lambda l: labels_dict.get(l))
    scores_df['score'] = scores_df['score'].apply(lambda x: round(x, 2))

    fig = st.bar_chart(data=scores_df, x='label', y='score',  width=300, height=300)

    scores_df = scores_df.sort_values(by='score', ascending=False)
    scores_df