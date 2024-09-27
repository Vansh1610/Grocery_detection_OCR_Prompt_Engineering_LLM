import streamlit as st
from PIL import Image
from ocr import test_image
import cv2
import numpy as np
from prompt_refine import prompt_refine
import re

st.set_page_config(layout="wide")

css = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-color: #089c34;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebarContent"] {
        color: black;
        background-color: white;
    }
    .st-emotion-cache-qcpnpn {
    background-color:white

    }
    h1{
    color:#089c34
    }

    .stSidebar{
            box-shadow: 3px 3px 5px 12px rgba(0,0,0,0.1);
            border-radius:1rem
    }

    .st-an{
    background-color: white;
    color:black
    }
    .st-emotion-cache-doy61h{
    color:black
    }
    .e1811lun0 {
    background-color: #089c34; 
    color: white;
    }

    .e1811lun0:not(.e1811lun0-selected) {
    background-color: white;
    color: black;
    }

    .e1811lun0:hover{
    background-color:#089c34;
    color:white

    }


    .st-emotion-cache-uef7qa {
    color:black
    }


    [data-testid="stFileUploaderDropzone"] {
    background-color:white;
    color:black
    }
    [data-testid="stBaseButton-secondary"]{
    color:white
    }
    [data-testid="stSliderThumbValue"]{
    color:#089c34
    }

    .st-emotion-cache-1dj3ksd {
    color:#089c34

    }

    .e1nzilvr5{

    color:black
    }

    .title{

    color:white
    }
    .custom-div{
    color: white;
    font-size:14px
    }

    .results-div{
    font-size:14px;
    color:white;
    margin-bottom:1rem
    }
    .st-emotion-cache-15hul6a {
    background-color:white;
    border-color:black;
    color:black
    }

    .st-emotion-cache-15hul6a:hover,.st-emotion-cache-15hul6a:active{

    border-color:black;
    color:black
    }

    img{

    width:400px
    }
    </style>
    """


st.markdown(css, unsafe_allow_html=True) 


st.markdown('<h1 class="title">Grocery App</h1>', unsafe_allow_html=True)

st.sidebar.title('Parameters')

threshold_method = st.sidebar.selectbox("Select Thresholding Method", ["Adaptive", "Normal", "Otsu"])

if threshold_method == "Normal":
    normal_threshold_value = st.sidebar.slider("Normal Threshold Value", 0, 255, 128)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 0, 300, 100)
display_boxes = st.sidebar.checkbox("Display Bounding Boxes", value=True)

col1, col3 = st.columns([0.4, 0.3], gap='large')

def refine_op(output):
    output = re.sub(r"\[", "", output)
    output = re.sub(r"]", "", output)
    output = re.sub(r"\n", ",", output)

    refine = output.split(',')
    i = 0
    output = []
    while i < len(refine):
        var = ' '.join(refine[i:i+3])
        var = re.sub(r'"', "", var)
        var = re.sub(r"'", "", var)
        output.append(var)
        i += 3

    return output

with col1:
    st.markdown('<div class="custom-div">Upload an Ingredient List Image</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("",type=['jpg', 'jpeg', 'png'])

    if uploaded_image:
    

        image = Image.open(uploaded_image)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner('Processing image...'):
            if threshold_method == "Normal":
                output, processed_image = test_image(img_bgr, threshold_method, min_contour_area, display_boxes, normal_threshold_value)
            else:
                output, processed_image = test_image(img_bgr, threshold_method, min_contour_area, display_boxes, 0)

            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb)
            output = prompt_refine(output)
            output = refine_op(output)

with col3:
    if uploaded_image:
        st.markdown('<div class="results-div">Results</div>', unsafe_allow_html=True)
        for i in output:
            con = st.container(border=True)
            con.write(i)
