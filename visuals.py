import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(df):
    g = sns.lmplot(
        x='step',
        y='loss_scores',
        data=df,
        col='model',
        hue='model',
        col_wrap=2,
        fit_reg=False
    )
    g.set_axis_labels("Steps", "Loss")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Loss")

    return g


def plot_accuracy(df):
    g = sns.lmplot(
        x='step',
        y='acc_scores',
        data=df,
        col='model',
        hue='model',
        col_wrap=2,
        fit_reg=False
    )
    g.set_axis_labels("Steps", "Accuracy")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Accuracy")

    return g
