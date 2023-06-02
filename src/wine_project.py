
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from chatgpt_tensorflow_03 import machine_learn


# Tools ------------------------------------------------------------------

def sort_val_vec(values: np.ndarray, matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    order = np.argsort(values)[::-1]
    return [values[i] for i in order], [matrix[i] for i in order]


def pca(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    cov = df.cov()
    e_vals, e_vecs = np.linalg.eig(cov)
    e_vecs = np.transpose(e_vecs)
    return sort_val_vec(e_vals, e_vecs)


def normalize_array(array: np.ndarray) -> np.ndarray:
    return (array - np.mean(array)) / np.std(array)


# Functions ------------------------------------------------------------------

def get_wine_df() -> pd.DataFrame:
    os.chdir(r"../data")
    return pd.read_csv("winequality-red.csv", delimiter=';')


def remove_high_outliers(df: pd.DataFrame) -> pd.DataFrame:
    upper_limit_dict = {'fixed acidity': 14.5,
                        'volatile acidity': 1.15,
                        'citric acid': 0.8,
                        'residual sugar': 10,
                        'chlorides': 0.5,
                        'free sulfur dioxide': 60,
                        'total sulfur dioxide': 200,
                        'density': None,
                        'pH': 3.8,
                        'sulphates': 1.4,
                        'alcohol': 14.5}
    high_outliers_indices = []
    for key, value in upper_limit_dict.items():
        indices = df[df[key] > value].index
        for i in indices:
            if i not in high_outliers_indices:
                high_outliers_indices.append(i)
    return df.drop(high_outliers_indices)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize all columns using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    # Create a new DataFrame with normalized values
    return pd.DataFrame(normalized_data, columns=df.columns)


def plot_all(df: pd.DataFrame, show_separate: bool = False) -> None:
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.title("All Attributes")
    plt.legend()
    plt.show()
    if not show_separate:
        return
    for attribute in df.columns:
        plt.plot(df[attribute])
        plt.title(attribute)
        plt.xlabel("index")
        plt.ylabel(attribute)
        plt.show()
        plt.close()


def sort_attributes(df: pd.DataFrame) -> None:
    for column in df.columns:
        plt.plot(np.sort(df[column]), label=column)
    plt.legend()
    plt.show()

    high_vals = ['free sulfur dioxide', 'total sulfur dioxide']
    medium_vals = ['fixed acidity', 'residual sugar', 'pH', 'alcohol', 'quality']
    low_vals = ['volatile acidity', 'citric acid', 'chlorides', 'density', 'sulphates']
    titles = ["Low Vals", "Medium Vals", "High Vals"]
    all_vals = [low_vals, medium_vals, high_vals]
    maxes = [2, 15, 280]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    for title, maxi, vals, ax in zip(titles, maxes, all_vals, (ax1, ax2, ax3)):
        for name in vals:
            ax.plot(np.sort(df[name]), label=name)
        # plt.vlines(1500, 0, maxi, 'k')
        ax.set_title(title)
        ax.legend()
    plt.show()


def plot_against_quality(df: pd.DataFrame) -> None:
    n_cols = 6
    fig, axs = plt.subplots(2, n_cols, sharey="all")
    fig.suptitle("First 9 components vs quality, with linfit line")
    for i, attribute in enumerate(df.columns):
        if attribute == 'quality':
            continue
        ax = axs[i // n_cols][i % n_cols]
        ax.scatter(df[attribute], df['quality'])
        ax.set_xlabel(attribute)
        if not i % n_cols:
            ax.set_ylabel('quality')
        coef = np.polyfit(df[attribute], df['quality'], 1)
        min_at = min(df[attribute])
        max_at = max(df[attribute])
        xs = np.linspace(min_at, max_at, 2000)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, color='r')
        # ax.set_title(f"a = {coef[1]:.2f}, b = {coef[0]:.2f}")
        # print(f"{attribute} & {coef[0]:.4f} \\\\")
        print(f"{coef[0]:.4f}")
    plt.show()


def pca_table(df: pd.DataFrame, show_scree: bool = True) -> None:
    vals, vecs = pca(df)
    print("Eigenvalue", end="")
    for val in vals:
        print(f" & {val:.2f}", end="")
    print(" \\\\")

    sum_val = sum(vals)
    fracs = tuple(100 * val / sum_val for val in vals)
    print("Percentage(%)", end="")
    for frac in fracs:
        print(f" & {frac:.2f}", end="")
    print(" \\\\")

    for i, attribute in enumerate(df.columns):
        print(attribute, end="")
        for vec in vecs:
            print(f" & {round(1000*vec[i])}", end="")
        print(" \\\\")

    if show_scree:
        plt.plot(fracs, color='b')
        for i, frac in enumerate(fracs):
            plt.scatter(i, frac, color='b', label=f"{frac:.2f}%")
        plt.title("Scree plot of PCA")
        plt.legend()
        plt.xlabel("Principal Component")
        plt.ylabel("Variance explained")
        plt.show()


def scatter_first_2_components(clipped_df: pd.DataFrame, normalized_df: pd.DataFrame) -> None:
    vals, vecs = pca(normalized_df)
    transform_matrix = np.transpose(vecs[:2])
    new_data = np.dot(normalized_df, transform_matrix)
    colors = ['k', 'k', 'k', 'r', 'g', 'b', 'y', 'm', 'c']
    color_lis = [colors[i] for i in clipped_df['quality']]
    plt.scatter(new_data[:, 0], new_data[:, 1], c=color_lis)
    for i in range(3, 9):
        plt.scatter(-4, -4, c=colors[i], label=i)
    plt.scatter(-4, -4, s=72, c='w')
    plt.title("Scatter plot of first 2 principal components with quality as color")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()


def remove_2_dimensions(df: pd.DataFrame, return_df: bool = False) -> np.ndarray | pd.DataFrame:
    vals, vecs = pca(df)
    transform_matrix = np.transpose(vecs[:-2])
    dimension_reduced_array = np.dot(df, transform_matrix)
    if not return_df:
        return dimension_reduced_array
    else:
        df = pd.DataFrame(dimension_reduced_array)
        df.columns = [f"component {i+1}" for i in range(len(vals)-2)]
        return df


def plot_ml_predicted(clipped_df: pd.DataFrame) -> None:
    weights = [0.05973548, -1.027153, -0.2547749, 0.0045553, -0.93602085, 0.00216343,
               -0.00248034, 0.32539114, 0.35694644, 1.0199566, 0.3231147]
    bias = 0.4254112
    testing_loss = 0.45
    predicted_qualities = np.dot(clipped_df.iloc[:, :11], np.transpose(weights)) + bias
    plt.scatter(clipped_df["quality"], predicted_qualities, label="scatter")
    plt.gca().set_aspect("equal")
    xs = [3, 8]
    plt.plot(xs, xs, color='k', label="y=x")
    plt.title("real vs predicted qualities, of ML")
    plt.ylabel("predicted qualities")
    plt.xlabel("real qualities")
    coef = np.polyfit(clipped_df["quality"], predicted_qualities, 1)
    plt.plot(xs, np.polyval(coef, xs), color='r', label=f"fit, testing loss: {testing_loss}")
    plt.legend()
    plt.show()


def plot_ml_dimension_reduced(dimension_reduced_df: pd.DataFrame):
    weights = [[-0.05233626], [0.2821826], [-0.19712853], [0.02563123], [-0.0879453],
               [-0.01019083], [-0.09990207], [0.068753], [0.12720628]]
    bias = 5.6673555
    testing_loss = 0.43
    predicted_qualities = np.dot(dimension_reduced_df.iloc[:, :9], weights) + bias
    plt.scatter(dimension_reduced_df["quality"], predicted_qualities, label="scatter")
    plt.gca().set_aspect("equal")
    xs = [3, 8]
    plt.plot(xs, xs, color='k', label="y=x")
    plt.title("real vs predicted qualities, of ML on dimension reduced input")
    plt.ylabel("predicted qualities")
    plt.xlabel("real qualities")
    coef = np.polyfit(dimension_reduced_df["quality"], predicted_qualities, 1)
    plt.plot(xs, np.polyval(coef, xs), color='r', label=f"fit, testing loss: {testing_loss}")
    plt.legend()
    plt.show()


def plot_lin_fit_predicted(dimension_reduced_df: pd.DataFrame):
    weights = [-0.0644, 0.2711, -0.2079,  0.0266, -0.0771,
               -0.0098, -0.1128, 0.0844, 0.1159]
    bias = 5.6673555
    testing_loss = "?"

    predicted_qualities = np.dot(dimension_reduced_df.iloc[:, :9], np.transpose(weights)) + bias
    plt.scatter(dimension_reduced_df["quality"], predicted_qualities, label="scatter")
    plt.gca().set_aspect("equal")
    xs = [3, 8]
    plt.plot(xs, xs, color='k', label="y=x")
    plt.title("real vs predicted qualities, of lin-fit on dimension reduced input")
    plt.ylabel("predicted qualities")
    plt.xlabel("real qualities")
    coef = np.polyfit(dimension_reduced_df["quality"], predicted_qualities, 1)
    plt.plot(xs, np.polyval(coef, xs), color='r', label=f"fit, testing loss: {testing_loss}")
    plt.legend()
    plt.show()


def scatter_2_highest_impact(df: pd.DataFrame) -> None:
    colors = ['k', 'k', 'k', 'r', 'g', 'b', 'y', 'm', 'c']
    color_list = [colors[q] for q in df['quality']]
    plt.scatter(df[:, 1], df[:, 2], c=color_list)
    for i in range(3, 9):
        plt.scatter(-4, -4, c=colors[i], label=i)
    plt.scatter(-4, -4, s=72, c='w')
    plt.title("Scatter plot of first 2 principal components with quality as color")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()


# Run ------------------------------------------------------------------

def main():
    # cleaning data ------------------------------------------------------------------
    initial_wine_df = get_wine_df()
    clipped_wine_df = remove_high_outliers(initial_wine_df)
    clipped_qualities_array = clipped_wine_df["quality"].values
    normalized_wine_df = normalize_df(clipped_wine_df)
    dimension_reduced_wine_df = remove_2_dimensions(normalized_wine_df.iloc[:, :11], return_df=True)    # , return_df=True
    dimension_reduced_wine_df['quality'] = pd.DataFrame(clipped_qualities_array)

    # tests ------------------------------------------------------------------
    # plot_all(initial_wine_df)
    # sort_attributes(initial_wine_df)
    # pca_table(clipped_wine_df.iloc[:, :11])
    # scatter_first_2_components(clipped_wine_df, normalized_wine_df)
    # plot_against_quality(dimension_reduced_wine_df)
    # plot_ml_predicted(clipped_wine_df)
    # machine_learn(dimension_reduced_wine_df, clipped_qualities_array)
    # plot_ml_dimension_reduced(dimension_reduced_wine_df)
    plot_lin_fit_predicted(dimension_reduced_wine_df)


if __name__ == "__main__":
    main()


# Old ------------------------------------------------------------------

def all_pcas(df: pd.DataFrame) -> None:
    vals, vecs = pca(df)
    for val, vec in zip(vals, vecs):
        plt.plot(vec, label=str(val))
    plt.legend()
    plt.title("All PCA's")
    plt.show()


def pca_number(df: pd.DataFrame, pca_n: int = 0) -> None:
    vals, vecs = pca(df)
    plt.plot(vecs[pca_n]*1000)
    plt.title("First PCA")
    plt.show()
    for i, j in zip(vecs[pca_n]*1000, df.columns):
        print(j, i, sep=" & ")

