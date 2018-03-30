import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")


def plot_loss(df, save=False):
    g = sns.lmplot(
        x='step',
        y='loss_scores',
        data=df,
        hue='epoch',
        fit_reg=False
    )
    g.set_axis_labels("Steps", "Loss")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Loss")
    g.fig.figsize = (12,9)

    if save:
        g.savefig('plots/loss.png', dpi=256)

    return g


def plot_accuracy(df, save=False):
    g = sns.lmplot(
        x='step',
        y='acc_scores',
        data=df,
        hue='epoch',
        fit_reg=False
    )
    g.set_axis_labels("Steps", "Accuracy")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Accuracy")

    if save:
        g.savefig('plots/accuracy.png', dpi=256)

    return g
