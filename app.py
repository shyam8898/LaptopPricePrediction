import numpy as np
import streamlit as st
import pickle
import sklearn

#import the model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Prediction")

# brand
company=st.selectbox('Brand',df['Company'].unique())

# type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

# ram
ram=st.selectbox('RAM(in GB)',df['Ram'].unique())

# weight
weight=st.number_input('Weight(in Kg)')

# touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

# ips
ips=st.selectbox('IPS',['No','Yes'])

# ppi
# 1) screensize
screensize=st.number_input('Screensize(in Inches)')
# 2) resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu=st.selectbox('CPU',df['Cpu brand'].unique())

# hdd
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# ssd
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# gpu
gpu=st.selectbox('GPU',df['Gpu brand'].unique())

# os
os=st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0
    X_res=int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi=((X_res**2) + (Y_res**2))**0.5/screensize
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title("The Predicted Price For This Configuration Is Rs. " + str(int(np.exp(pipe.predict(query)[0]))))




