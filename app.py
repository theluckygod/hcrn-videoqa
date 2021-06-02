import streamlit as st
from requests import post
import yaml
from os.path import join
import base64

@st.cache
def load_conf(conf_path):
    with open(conf_path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    return cfg

app_conf_path = "./app_config.yml"
app_conf = load_conf(app_conf_path)

# Side bar
st.sidebar.subheader('Dataset format')
data_format_values = app_conf['data_format_values']
data_format_default = app_conf['data_format_default']
data_format = st.sidebar.selectbox('', data_format_values, index=data_format_values.index(data_format_default))
        
cfg = load_conf(join(app_conf['conf_folder'], data_format + '.yml'))

# Content
st.title("Video Question Answering")
        
f = st.file_uploader("Upload a video (.mp4)",type=['mp4'])
if f is not None:
    video_data = f.getvalue()
    st.video(video_data)
    video_data_encoded = base64.b64encode(video_data)

    default_value_text_area='What are they doing?'
    sentence = st.text_area(label='Input your question here:',
                            value=default_value_text_area,
                            height=None) 
    
    if st.button("Ask"):
        data = {'ques': sentence, 'data_type': data_format, 'video': video_data_encoded}
        print("\n------------------------------------")
        temp_d = {'ques': sentence, 'data_type': data_format, 'video': f.name}
        print(f"POST {temp_d}")

        is_reponse = False
        try:
            reponse = post('http://localhost:5000/inference', data=data).json()
            is_reponse = True
            print("_____Response____")
        except:
            print("_____No response____")

        if is_reponse:
            try:
                st.text(f"Answer: {reponse['answer']}")
                print("Writing the answer successfully")
            except:
                print("Cannot write the answer!!!")
        else:
            st.text('No response from server.')
            