import whisper
from pytube import YouTube
import requests, io
from urllib.request import urlopen
from PIL import Image
import time
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import os

st.set_page_config(page_title="YouTube Transcriber", page_icon="🗣", layout="wide")


# Define a function that we can use to load lottie files from a link.
@st.cache(allow_output_mutation=True)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

col1, col2 = st.columns([1, 3])
with col1:
    lottie = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_bntlaz7t.json")
    st_lottie(lottie, speed=1, height=200, width=200)

with col2:
    st.write("""
    ## Youtube Transcriber 
    ##### This is an app that transcribes YouTube videos into text.""")


#def load_model(size):
    #default_size = size
    #if size == default_size:
        #return None
    #else:
        #loaded_model = whisper.load_model(size)
        #return loaded_model 
    

@st.cache(allow_output_mutation=True)
def populate_metadata(link):
    yt = YouTube(link)
    author = yt.author
    title = yt.title
    description = yt.description
    thumbnail = yt.thumbnail_url
    length = yt.length
    views = yt.views
    return author, title, description, thumbnail, length, views

# Uncomment if you want to fetch the thumbnails as well.
#def fetch_thumbnail(thumbnail):
    #tnail = urlopen(thumbnail)
    #raw_data = tnail.read()
    #image = Image.open(io.BytesIO(raw_data))
    #st.image(image, use_column_width=True)


def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


loaded_model = whisper.load_model("base")
current_size = "None"
size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large"], index=1)


def change_model(current_size, size):
    if current_size != size:
        loaded_model = whisper.load_model(size)
        st.write(f"Model is {'multilingual' if loaded_model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in loaded_model.parameters()):,} parameters.")
        return loaded_model
    else:
        return None


@st.cache(allow_output_mutation=True)
def inference(link):
    yt = YouTube(link)
    path = yt.streams.filter(only_audio=True)[0].download(filename="audio.mp4")
    results = loaded_model.transcribe(path)
    return results["text"]


def main():
    change_model(current_size, size)
    link = st.text_input("YouTube Link")
    if st.button("Transcribe"):
        author, title, description, thumbnail, length, views = populate_metadata(link)
        results = inference(link)
            
        col3, col4 = st.columns(2)
        with col3:
            #fetch_thumbnail(thumbnail)
            st.video(link)
            st.markdown(f"**Channel**: {author}")
            st.markdown(f"**Title**: {title}")
            st.markdown(f"**Length**: {convert(length)}")
            st.markdown(f"**Views**: {views:,}")

        with col4:
            with st.expander("Video Description"):
                st.write(description)
            #st.markdown(f"**Video Description**: {description}")
            with st.expander("Video Transcript"):
                st.write(results)
            # Write the results to a .txt file and download it.
            with open("transcript.txt", "w+") as f:
                f.writelines(results)
                f.close()
            with open(os.path.join(os.getcwd(), "transcript.txt"), "rb") as f:
                data = f.read()
                if st.download_button(label="Download Transcript",
                                data=data,
                                file_name="transcript.txt"):
                    st.success("Downloaded Successfully!")

if __name__ == "__main__":
    main()