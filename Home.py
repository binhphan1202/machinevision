import streamlit as st

st.set_page_config(
    page_title="Home Page",
    page_icon='🏠'
)
header = '<p style="font-family: Courier; color: #cc9900; font-size: 40px; text-align: center;"><b>Machine Vision</b></p>'
st.markdown(header, unsafe_allow_html=True)
st.markdown('<p style="font-family: Bradley Hand, cursive; text-align: center; color: #800000; font-size: 38px;">Final Project</p>', 
            unsafe_allow_html=True)

with st.columns(3)[1]:
    st.image('hcmute.png')

heading = '<p style="font-family: helvetica, san-serif; color: #ffff80; font-size: 22px;"><b>INFORMATION</b></p>'
st.sidebar.markdown(heading, unsafe_allow_html=True)
mem1 = '<p style="font-family: trebuchet ms, san-serif; font-size: 16px; color: #000000;"><i>Phan Le Thanh Binh-20146149</i></p>'
st.sidebar.markdown(mem1, unsafe_allow_html=True)
mem2 = '<p style="font-family: trebuchet ms, san-serif; font-size: 16px; color: #000000;"><i>Ha Thanh Binh-20146304</i></p>'
st.sidebar.markdown(mem2, unsafe_allow_html=True)
