#Library imports
import torch
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from torchvision import transforms



model=torch.load('model_cpu_2')
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
            if(pred==1):
                correctly_worn='YES'
        
        st.image(img.resize((256,256) , Image.ANTIALIAS))
        st.write("MASK WORN: ",mask_worn)
        if(mask_worn=='YES'):
            st.write("IS IT WORN CORRECTLY: ",correctly_worn)
