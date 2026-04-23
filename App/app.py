import streamlit as st
from pipeline.process import process_video

st.title("🌊 AquaVision AI")

uploaded_file = st.file_uploader("Upload underwater video")

if uploaded_file:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp.mp4")

    if st.button("Analyze"):
        df = process_video("temp.mp4")

        st.write("### Results")
        st.dataframe(df)

        st.line_chart(df["score"])