"""
Visualization Module
"""

import matplotlib.pyplot as plt

from ml.utils.config import NUMERICAL_PLOTS_DIR


def save_histogram(df, column):

    plt.figure(figsize=(8, 5))

    df[column].hist(bins=30)

    plt.title(column)

    plt.xlabel(column)

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(NUMERICAL_PLOTS_DIR / f"{column}_histogram.png")

    plt.close()