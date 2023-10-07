#Library imports
import torch
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from torchvision import transforms

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1436262513933-a0b06755c784?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTV8fHdlYnBhZ2UlMjBiYWNrZ3JvdW5kfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

model=torch.load('model_cpu_4')
model.eval()

st.write("# Wear your Mask....The Correct Way!")
st.write("Select an image of a person to know whether Face Mask is worn, if worn is it properly worn.")
image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
submit = st.button('Predict')

if submit:

    if image_file is not None:

        img=Image.open(image_file)
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )])

        img_preprocessed = preprocess(img)
        batch_img = torch.unsqueeze(img_preprocessed, 0)

        output=model(batch_img)
        value,pred=torch.max(output.data,1)
        pred=pred.item()
        mask_worn='NO'
        correctly_worn='NO'
        if (pred<2):
            mask_worn='YES'
            # if(pred==1):
            #     correctly_worn='YES'
        
        st.image(img.resize((256,256) , Image.ANTIALIAS))
        st.write("MASK WORN: ",mask_worn)
        # if(mask_worn=='YES'):
        #     st.write("IS IT WORN CORRECTLY: ",correctly_worn)
