import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(df, save=False):
    fig, ax = plt.subplots()
    epochs = df.groupby('epoch')
    for epoch, data in epochs:
        ax.plot(data['step'], data['loss_scores'], label=epoch)

    ax.legend(numpoints=1, loc='upper right')
    ax.set(
        xlabel="Steps",
        ylabel="Loss",
        title="Model Loss",
    )
    if save:
        fig.savefig('plots/loss.png', dpi=256)

    return fig, ax


def plot_accuracy(df, save=False):
    fig, ax = plt.subplots()
    epochs = df.groupby('epoch')
    for epoch, data in epochs:
        ax.plot(data['step'], data['acc_scores'], label=epoch)

    ax.legend(numpoints=1, loc='lower right')
    ax.set(
        xlabel="Steps",
        ylabel="Accuracy",
        title="Model Accuracy",
    )
    if save:
        fig.savefig('plots/accuracy.png', dpi=256)

    return fig, ax
