import streamlit as st
from typing import List, Callable, Tuple, NoReturn, Any
from pathlib import Path
import os
import random
import json

def set_title() -> NoReturn:
    st.markdown("<h1 style='text-align: center; color: white;'>Induction</h1>", unsafe_allow_html=True)


def main():
    set_title()
    attn_weights_dir = '/checkpoint/itayitzhak/attn_weigths/'
    #base_data_dir = Path("/home/olab/orhonovich/induction/data_generation/raw")
    base_data_dir = Path(attn_weights_dir)
    data_dir = st.selectbox("Tasks", options=["Select task"] + os.listdir(base_data_dir) )
    if data_dir != "Select task":
        data_file = st.selectbox("Select file to read", options=["Select file"] +os.listdir(base_data_dir / data_dir))
        if data_file != "Select file":
            with open(base_data_dir / data_dir / data_file, 'r') as f:
                data = json.load(f)
                st.write(data)

main()