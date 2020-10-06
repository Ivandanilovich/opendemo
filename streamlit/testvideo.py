import streamlit as st
import streamlit.components.v1 as components

components.html(

'''<video>
  <source src="alr_video1.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>'''
)

st.video()