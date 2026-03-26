import pandas as pd
import numpy as np
import os

def load_dataset():
    possible_paths = [
        "auction.csv",
        "bidding/auction.csv",
        "venv/bidding/auction.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)

            if "auctionid" in df.columns and "bid" in df.columns:
                grouped = df.groupby("auctionid")
                auctions = []

                for _, auction in grouped:
                    bids = auction["bid"].values
                    auctions.append({
                        "bids": bids,
                        "valuations": bids + np.random.uniform(0, 10, len(bids))
                    })

                return auctions

    return None
