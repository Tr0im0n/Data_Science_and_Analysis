
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

os.chdir(r"../data")
diamond_file_name = "diamonds.csv"
df = pd.read_csv(diamond_file_name, delimiter=',')
# column_names = tuple(df.columns)

str_columns = ('cut', 'color', 'clarity')
flt_columns = ('carat', 'depth', 'table', 'x', 'y', 'z')    # , 'price'

bar_options = tuple( tuple(df[column].unique()) for column in str_columns )
# print(bar_options)

list_tuple = tuple( df[column].tolist() for column in str_columns )
height_tuple = tuple( tuple( column.count(option) for option in bar_option ) for column, bar_option in zip(list_tuple, bar_options) )


def bar_plots():
    for column, options, bar_height in zip(str_columns, bar_options, height_tuple):
        plt.bar(options, bar_height)
        plt.title(column)
        plt.show()


def price_plot():
    plt.plot(df['price'])
    plt.title('price')
    plt.show()


def flt_plots():
    for column in flt_columns:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.show()


def carat_price_scatter():
    plt.scatter(df['carat'], df['price'], 1, label='scatter')
    plt.xlabel('carat')
    plt.ylabel('price')
    p = np.polyfit(df['carat'], df['price'], 1)
    x = np.linspace(0, 3, 3001)
    plt.plot(x, np.polyval(p, x), color='r', label='lin fit')
    plt.legend()
    plt.show()


bar_plots()
price_plot()
flt_plots()
carat_price_scatter()
