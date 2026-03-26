import streamlit as st
import numpy as np
import pandas as pd

from data_loader import load_dataset
from env import AuctionEnv
from train import train_model
from utils import simulate

st.set_page_config(page_title="AI Auction", layout="wide")

st.title("🧠 AI Auction Mechanism")
st.write("AI learns auction rules dynamically using RL")

auctions = load_dataset()

if auctions:
    st.success("Using real dataset")
else:
    st.warning("Dataset not found → using simulated data")

tab1, tab2 = st.tabs(["Simulation", "Analysis"])

with tab1:
    n_agents = st.slider("Bidders", 2, 20, 5)
    runs = st.slider("Simulations", 10, 200, 50)

    if st.button("Train & Run"):
        env = AuctionEnv(auctions=auctions, n_agents=n_agents)
        model = train_model(env)

        ai_list = []

        for _ in range(runs):
            ai, _, _, _, _, _ = simulate(env, model)
            ai_list.append(ai)

        st.bar_chart({"AI Revenue": np.mean(ai_list)})

with tab2:
    st.write("Analysis section")
