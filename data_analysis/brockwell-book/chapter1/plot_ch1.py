import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from datasets.data_load import load_dataset


def plot_series(data_file: str):
    data = load_dataset(data_file)
    print(data)
    fig, ax = plt.subplots()
    ax.plot(data)
    plt.savefig(fig, "./figure1-1.png")

def plot_figure_1_1():
    data = load_dataset("./datasets/data/wine.txt")
    data = np.array(data)
    values = data[:,2]
    dates = []
    for i in range(data.shape[0]):
        dates.append(datetime.date(int(data[i,0]), int(data[i,1]), 1))
    data_df = pd.DataFrame({"wine":values}, index=dates)
    print(data_df)
    fig, ax = plt.subplots()
    ax.plot(data_df, marker="s", color="black")
    plt.show()
    # ax.plot(years, values)
    # plt.savefig(fig, "./figure1-1.png")





if __name__ == "__main__":
   plot_figure_1_1() 
