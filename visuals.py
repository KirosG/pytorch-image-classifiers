import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('darkgrid')


def plot_loss(df, save=True):
    fig, ax = plt.subplots()
    # epochs = df.groupby('epoch')
    # for epoch, data in epochs:
    #     ax.plot(data['step'], data['loss_scores'], label='Epoch %d' % epoch)
    ax.plot(df.index, df['loss_scores'])

    # ax.legend(numpoints=1, loc='upper right')
    ax.set(xlabel="Steps", ylabel="Loss", title="Model Loss")
    if save:
        fig.savefig('plots/loss.png', dpi=256)

    return fig, ax


def plot_accuracy(df, save=False):
    fig, ax = plt.subplots()
    # epochs = df.groupby('epoch')
    # for epoch in df['epoch'].unique():
        # ax.plot(df[df['epoch'] == epoch].index, df[df['epoch'] == epoch]['acc_scores'], label='Epoch %d' % epoch)
    ax.plot(df.index, df['acc_scores'])
    # ax.legend(numpoints=1, loc='lower right')
    ax.set(xlabel="Steps", ylabel="Accuracy", title="Model Accuracy")
    if save:
        fig.savefig('plots/accuracy.png', dpi=256)

    return fig, ax
