import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
import os
import Chuong3 as c3
import Chuong4 as c4

st.set_page_config(
    page_title="Image Processing",
    page_icon="📸",
)
st.markdown("<h2 style = 'text-align: center; font-size: 40px; font-family: comic sans ms, cursive; color: #800000;'>Image Processing</h2>", unsafe_allow_html=True)
st.sidebar.header("Digital Image Processing")

def imgin_header():
    imgin_Title = '<p style="font-family: Courier; color: #3333ff; font-size: 26px;">Original Uploaded Image</p>'
    st.markdown(imgin_Title, unsafe_allow_html=True)
def imgout_header():
    imgout_Title = '<p style="font-family: Courier; color: #3333ff; font-size: 26px;">Image after being processed</p>'
    st.markdown(imgout_Title, unsafe_allow_html=True)
def download_info():
    download_notice = '<p style="font-family: monospace; font-size: 14px; color: #c2c2d6;"><i>For downloading, please right click on the image.</i></p>'
    st.markdown(download_notice, unsafe_allow_html=True) 
    
option = st.sidebar.selectbox('Choose topic', ['--Select topic--', 'Chương 3', 'Chương 4'])
if option == 'Chương 3':
    img_upload = st.file_uploader('Upload Image', type=['jpg', 'tif', 'bmp', 'gif', 'png'])
    if img_upload is not None:
        # option = st.sidebar.selectbox('Choose topic', ['Chương 3', 'Chương 4'])
        # if option == 'Chương 3':
            global imgin
            filepath = 'D:\BINH\Year_3\Sem_2\Machine_Vision\ProjectCuoiKy\ProcessingImage\Chuong3\\' + img_upload.name
            imgin = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            imgin_header()
            st.image(imgin)

            c3_function = st.sidebar.selectbox('Please pick up a function', [
                '--Select function--','Negative', 'Logarit', 'Power', 'PiecewiseLinear', 'IRange', 'BitPlane', 'Histogram',
                'HistEqual', 'HistEqualColor', 'LocalHist', 'HistStat', 'BoxFilter',
                'GaussFilter', 'Smooth'])
            # Negative
            if c3_function == 'Negative':
                global imgout
                imgout = c3.Negative(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # Logarit
            elif c3_function == 'Logarit':
                imgout = c3.Logarit(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # Power
            elif c3_function == 'Power':
                imgout = c3.Power(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # PiecewiseLinear
            elif c3_function == 'PiecewiseLinear':
                imgout = c3.PiecewiseLinear(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # IRange
            elif c3_function == 'IRange':
                imgout = c3.IRange(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # BitPlane
            elif c3_function == 'BitPlane':
                imgout = c3.BitPlane(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # Histogram
            elif c3_function == 'Histogram':
                imgout = c3.Histogram(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # HistEqual            
            elif c3_function == 'HistEqual':
                imgout = c3.HistEqual(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # HistEqualColor
            elif c3_function == 'HistEqualColor':
                imgout = c3.HistEqualcolor(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # LocalHist
            elif c3_function == 'LocalHist':
                imgout = c3.LocalHist(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # HistStat
            elif c3_function == 'HistStat':
                imgout = c3.HistStat(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # BoxFilter
            elif c3_function == 'BoxFilter':
                imgout = c3.BoxFilter(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # GaussFilter
            elif c3_function == 'GaussFilter':
                imgout = c3.GaussFilter(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            # Smooth
            elif c3_function == 'Smooth':
                imgout = c3.Smooth(imgin)
                imgout_header()
                st.image(imgout)
                download_info()
            else:
                st.image('waiting.gif', caption='Wait for processing...')

elif option == 'Chương 4':
    img_upload = st.file_uploader('Upload Image', type=['jpg', 'tif', 'bmp', 'gif', 'png'])
    if img_upload is not None:
        filepath = 'D:\BINH\Year_3\Sem_2\Machine_Vision\ProjectCuoiKy\ProcessingImage\Chuong4\\' + img_upload.name
        imgin = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        imgin_header()
        st.image(imgin)

        c4_function = st.sidebar.selectbox('Please pick up a function', [
            '--Select function--','Spectrum', 'FrequencyFilter', 'DrawNotchRejectFilter', 'NotchRejectFilter'])
        # Spectrum
        if c4_function == 'Spectrum':
            imgout = c4.Spectrum(imgin)
            imgout_header()
            st.image(imgout)
            download_info()
        # FrequencyFilter
        elif c4_function == 'FrequencyFilter':
            imgout = c4.FrequencyFilter(imgin)
            imgout_header()
            st.image(imgout)
            download_info()
        # DrawNotchRejectFilter
        elif c4_function == 'DrawNotchRejectFilter':
            imgout = c4.DrawNotchRejectFilter()
            imgout_header()
            st.image(imgout)
            download_info()
        # NotchRejectFilter
        elif c4_function == 'NotchRejectFilter':
            imgout = c4.NotchRejectFilter(imgin)
            imgout_header()
            st.image(imgout)
            download_info()
        else:
            st.image('waiting.gif', caption='Wait for processing...')
else:
    st.image('annoucement.png', use_column_width=True)



        