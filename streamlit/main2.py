import base64

import streamlit as st

# video_file = open('C:/Users/ivand/Downloads/Brickleberry Seasons 2/Brickleberry.S02E01.Miracle.Lake.720p.WEB-DL.x264.AAC.mp4', 'rb')
# video_file = open('C:/Users/ivand/Downloads/videod (12).avi', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.write('<h1></h1>')

#


import streamlit.components.v1 as components

# bootstrap 4 collapse example
components.html(
    """
    <a href=".avi" download=".avi">Download Your Expense Report</a>
    """,
    # height=600,
)

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

# st.markdown(get_table_download_link(df), unsafe_allow_html=True)
st.markdown('<a href=".avi"  target="_blank">Download</a>', unsafe_allow_html=True)
