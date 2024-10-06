import streamlit as st
import pickle
import numpy as np

df=pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

st.title("Laptop Price Predictor App")
st.write("This app is created using the data available on kaggle")
st.write("The prediction here is based on the data available, hence might not align with the real-life data")

company=st.selectbox("Manufacturing Company",df['Company'].unique(),index=4)
typename=st.selectbox("Type of laptop",df['TypeName'].unique(),index=1)
cpu=st.selectbox("Processor",df['Cpu'].unique(),index=0)
ram=st.radio("RAM(in GB)",[4,6,8,12,16,24,32,64,128],index=2,horizontal=True)
gpu=st.selectbox("Graphics Card",df['Gpu'].unique(),index=1)
os=st.selectbox("Operating System",df['OpSys'].unique(),index=2)
weight=st.slider('Weight(in kg)',min_value=0.7,max_value=4.7,value=2.0,step=0.1)
ips=st.radio("IPS Display?",['Yes','No'],horizontal=True)
touchscreen=st.radio("Touchscreen?",['Yes','No'],horizontal=True)
cpu_speed=st.slider("CPU clock speed",min_value=0.9,max_value=3.6,value=2.5,step=0.1)
hdd=st.radio("Hard drive capacity(in GB)[If SSD, set HDD value to 0]",[0,128,500,1000,2000],horizontal=True)
ssd=st.radio("SSD capacity(in GB)",[0,8,16,32,64,128,256,512,1000,2000],horizontal=True,index=7)
screen_size=st.slider("Screen size(measured in inches, calculated diagonally)",min_value=10.0,max_value=18.5,value=15.6,step=0.1)
screen_res=st.selectbox("Screen Resolution",["1366x768","1440x900","1600x900","1920x1080","1920x1200",
                                             "2160x1440","2256x1504","2304x1440","2400x1600","2560x1440",
                                             "2560x1600","2736x1824","2880x1800","3200x1800","3840x2160"],index=3)
if st.button("PREDICT PRICE"):
    ppi=None
    if (ips=='Yes'):
        ips=1
    else:
        ips=0
    if (touchscreen=="Yes"):
        touchscreen=1
    else:
        touchscreen=0
    X_res=int(screen_res.split('x')[0])
    Y_res=int(screen_res.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size

    query=np.array([[company,typename,cpu,ram,gpu,os,weight,ips,touchscreen,cpu_speed,hdd,ssd,ppi]])
    op=pipe.predict(query)
    f_op=int(round(op[0],-2))
    st.subheader("The estimated price of the laptop with the above selected configuration is ₹"+str(f_op))
