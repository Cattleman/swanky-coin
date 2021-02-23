'''
Placeholder app for NextSaga.io
'''

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.graph_objs as go
#import plotly.figure_factory as ff
import sys
sys.path.append('..')
#sys.path.append('../models')
from models.big_gan import gen_examples

def main():
    #st.title("NextSaga.io")

    page = st.sidebar.selectbox("Select a page", ["Home", "About", "Resources"])

    if page == "Home":
        #st.subheader("Home")
        st.markdown("## Welcome to NFT GAN!")

        st.write("These pictures are created with generative models!")
        image_out = gen_examples()
        st.image(image_out, use_column_width=True)
        #st.markdown("Here are some tools to help you tell your story.")
        # TODO include Yara welcome here
        
        st.write("You can purchase a NFT Token to own a random seed of the GAN!")

        # TODO


    if page == "About":
        st.subheader("About")

        st.markdown("Section is - WIP")

# Helpers
# TODO - add way to load the model only once with caching

if __name__ == '__main__':
    main()
