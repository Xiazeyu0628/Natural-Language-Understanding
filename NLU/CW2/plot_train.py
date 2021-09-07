import sys
from laplotter import LossAccPlotter
model = "/Users/xiazeyu/nlu_cw2/results/Q7/log.out"

plotter = LossAccPlotter(title="Training loss and validation loss of transformer",
                         save_to_filepath=model + ".png",
                         show_regressions=False,
                         show_averages=False,
                         show_loss_plot=True,
                         show_acc_plot=False,
                         show_plot_window=True,
                         x_label="Epoch")

with open(model) as f:
    while True:
        train = f.readline().strip().split()
        if 'loss' not in train:
            if not train:
                break
            continue

        dev = f.readline().strip().split()
        epoch = int(train[train.index('Epoch')+1][:-1])
        train_loss = float(train[train.index('loss')+1])
        dev_loss = float(dev[dev.index('valid_loss')+1])
        plotter.add_values(epoch,
                   loss_train=train_loss,
                   loss_val=dev_loss)
        print (epoch, train_loss, dev_loss)

plotter.block()
