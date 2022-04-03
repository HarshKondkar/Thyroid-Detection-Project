import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTEENN as sm
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

st.title('Think you have hypothyroid?')
st.header('Let us find out whether you need a check up!')
st.write('''Hypothyroidism is a common condition where the 
thyroid gland doesn't create and release enough thyroid hormone into your bloodstream. 
This makes your metabolism slow down. Also called underactive thyroid, hypothyroidism 
can make you feel tired, gain weight and be unable to tolerate cold temperatures.
Hypothyroidism's deficiency of thyroid hormones can disrupt such things as heart rate, body temperature and all aspects of metabolism. 
Hypothyroidism is most prevalent in older women.
Major symptoms include fatigue, cold sensitivity, constipation, dry skin and unexplained weight gain.
Treatment consists of thyroid hormone replacement.''')
st.markdown('**This app predicts your chances of having hypothyroid given your details.**')
st.markdown('**Please enter your information in the dialog box to your left.**')
st.sidebar.title('Enter your Info:')
img2 = Image.open('user_logo.png')
st.sidebar.image(img2)
dict_cols = {'Yes': 't',
             'No': 'f'}
img = Image.open('hypo_img.jpg')
st.image(img, caption = 'Symptoms of hypothyroid')

def user_input():
    age = st.sidebar.slider('Select your Age:', 1, 100, 35)
    sex = st.sidebar.selectbox('Select your gender:', ('M', 'F'))
    on_thyroxine = st.sidebar.selectbox('Are you on Thyroxine?', ('Yes', 'No'))
    query_on_thyroxine = on_thyroxine
    antithyroid = st.sidebar.selectbox('Are you on Antithyroid medication?', ('Yes', 'No'))
    sick = st.sidebar.selectbox('Do you feel sick occasionaly or are you sick right now?', ('Yes', 'No'))
    pregnant = st.sidebar.selectbox('Are you pregnant?', ('Yes', 'No'))
    thyroid_surgery = st.sidebar.selectbox('Have you undergone thyroid surgery?', ('No', 'Yes'))
    I131_treatment = st.sidebar.selectbox('Are you under the I131 treatment?', ('Yes', 'No'))
    query = st.sidebar.selectbox('You are worried about?', ('hyperthyroid', 'hypothyroid'))
    if query == 'hypothyroid':
        hypo_query = 'Yes'
        hyper_query = 'No'
    else:
        hypo_query = 'No'
        hyper_query = 'Yes'
    lithium = st.sidebar.selectbox('Are you taking Lithium?', ('Yes', 'No'))
    goitre = st.sidebar.selectbox('Are you diagnosed with goitre?', ('Yes', 'No'))
    tumor = st.sidebar.selectbox('Do you have a tumor?', ('Yes', 'No'))
    hypopituitary = st.sidebar.selectbox('Are you suffering from hypopituitarism?', ('Yes', 'No'))
    psych = st.sidebar.selectbox('Would your doctor define your psychiatric state as Good?', ('Yes', 'No'))
    TSH = st.sidebar.slider('Select your TSH level:', 0.0, 150.0, 100.0, step = 0.01)
    T3 = st.sidebar.slider('Select your T3 level:', 0.0, 11.0, 5.5, step = 0.01)
    TT4 = st.sidebar.slider('Select your TT4 level:', 0.0, 150.0, 50.0, step = 0.01)
    T4U = st.sidebar.slider('Select your T4U level:', 0.0, 3.0, 1.5, step = 0.01)
    FTI = st.sidebar.slider('Select your FTI level:', 0.0, 150.0, 50.0, step = 0.01)
    referral_source = st.sidebar.selectbox('Please mention your referral source:', ('SVI', 'SVHC', 'STMW', 'SVHD', 'Other'))
    data = {'age': age,
            'sex': sex,
            'on thyroxine': on_thyroxine,
            'query on thyroxine': query_on_thyroxine,
            'on antithyroid medication': antithyroid,
            'sick': sick,
            'pregnant': pregnant,
            'thyroid surgery': thyroid_surgery,
            'I131 treatment': I131_treatment,
            'query hypothyroid': hypo_query,
            'query hyperthyroid': hyper_query,
            'lithium': lithium,
            'goitre': goitre,
            'tumor': tumor,
            'hypopituitary': hypopituitary,
            'psych': psych,
            'TSH': TSH,
            'T3': T3,
            'TT4': TT4,
            'T4U': T4U,
            'FTI': FTI,
            'referral source': referral_source}
    features = pd.DataFrame(data, index = [0])
    return features
df = user_input()
cols_to_map = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', \
               'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', \
               'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
for col in cols_to_map:
    df[col] = df[col].map(dict_cols)
pipe = joblib.load('pipe.joblib')
pred = pipe.predict_proba(df)
if pred[0,1] >= 0.5:
    st.subheader('It is highly likely (**{}%**) that you have hypothyroid & should get yourself checked.'.format(np.round(pred[0,1]*100, 4)))
else:
    st.subheader('The chances of you having hypothyroid are low: **{}%**'.format(
        np.round(pred[0, 1]*100, 4)))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass