import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from models.WGAN_Generator import Generator
from models.WGAN_Generator_1 import Generator as w_gan
from models.DCGAN_Generator_0 import DCGAN_Generator_Model_0 as f_dcgan
from models.DCGAN_Generator_1 import DCGAN_Generator_Model_1 as s_dcgan 

st.header("Face Generation")
st.subheader("")

to_image = transforms.ToPILImage() #TODO put to the main function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_image_full_model(model_class, path):   
    model = model_class
    model = torch.load(path)
    print(model)
    model.eval()
    noise =  torch.randn(1, 100, 1, 1)#, device=device)
    img = model(noise).to(device)
    o_path = os.path.join('data','output','1.png')
    save_image(img, o_path, normalize=True)  
    img = torch.squeeze(img)
    img = to_image(img)
    
    return img, o_path

def generate_image_state_dict( model,path): 
    model = model.load_state_dict(torch.load(path ))
    model.eval()
    noise =  torch.randn(1, 100, 1, 1)
    img = model(noise)
    img = torch.squeeze(img)
    img = to_image(img)    

if st.button('Generate face with DCGAN Model 0'):
    model_class =  f_dcgan()
    path = os.path.join("data", "models", "dcgan_model_0.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width=256 , output_format='PNG')

if st.button('Generate face with DCGAN Model 1'):
    model_class =  s_dcgan()
    path = os.path.join("data", "models", "dcgan_model_1.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')

if st.button('Generate face with WGAN Model 0'):
    model_class = Generator()
    print(model_class)
    path = os.path.join("data", "models", "wgan_version_0.pt")
    image, o_path = generate_image_full_model(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')

if st.button('Generate face with WGAN Model 1'):
    model_class = w_gan()
    print(model_class)
    path = os.path.join("data", "models", "wgan_version_1.pt")
    image, o_path = generate_image_state_dict(model_class, path)
    st.image(o_path, width = 256, output_format='PNG')