import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
import pandas as pd

BATCH_KEYS_TO_PLOT = ["rot_l2_loss", "pos_l2_loss"]
# FRAME_KEYS_TO_PLOT = ["rot_diffs", "pos_diffs"]
FRAME_KEYS_TO_PLOT = []

# Function to close plot when "Esc" is pressed
def _on_key(event):
    if event.key == "escape":
        plt.close('all')

# Plots train and validation loss per epoch.
def plot_train_vs_val(base_dir, key, epochs):
    assert epochs is not None
    train_path = os.path.join(base_dir, "train", key)
    test_path = os.path.join(base_dir, "test", key)
    train_losses = []
    test_losses = []
    for epoch in epochs:
        train_csv_path = os.path.join(train_path, f"{key}_epoch_{epoch}.csv")
        test_csv_path = os.path.join(test_path, f"{key}_epoch_{epoch}.csv")
        train_loss = np.mean(pd.read_csv(train_csv_path)[key].to_numpy())
        test_loss = np.mean(pd.read_csv(test_csv_path)[key].to_numpy())
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event

    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, test_losses, label="test")

    epoch_step = len(epochs)//10 if len(epochs) > 10 else 1
    for i in range(0, len(epochs), epoch_step):
        x = epochs[i]
        tr_y = train_losses[i]
        te_y = test_losses[i]
        ax.annotate(f"{tr_y:.2f}", (x,tr_y))
        ax.annotate(f"{te_y:.2f}", (x,te_y))

    title = f"{key}: Train vs Validation"
    fig.suptitle(title)
    ax.set_xlabel("epoch #")
    ax.set_ylabel(f"mean {key}")
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(base_dir, f"{key}_train_vs_val.png"))
    plt.close(fig)


# Plots L2 losses in a dataset, with each point being the l2 loss for the batch.
# Plots multiple lines for each epoch.
def plot_per_batch(dir_path, key, epochs=None):
    if epochs is None:
        csv_path = os.path.join(dir_path, f"{key}.csv")
        dfs = pd.read_csv(csv_path)
    else:
        dfs = []
        for epoch in epochs:
            csv_path = os.path.join(dir_path, f"{key}_epoch_{epoch}.csv")
            dfs.append(pd.read_csv(csv_path))

    cmap = cm.get_cmap('viridis', len(epochs))  # Choose a colormap
    colors = [cmap(i) for i in range(len(epochs))]
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event

    if epochs is None:
        x = dfs['batch_idx']
        y = dfs[key]
        ax.plot(x, y)
    else:
        for epoch, df in enumerate(dfs):
            epoch = epoch+1  # 1-indexed
            x = df['batch_idx']
            y = df[key]
            ax.plot(x, y, label=f"epoch {epoch}", color=colors[epoch-1])

    title = f"{key}"
    fig.suptitle(title)
    ax.set_xlabel("batch #")
    ax.set_ylabel(key)
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(dir_path, f"{key}.png"))
    plt.close(fig)

# Only generates plot for a single epoch.
def plot_per_frame(dir_path, key, epoch=None):
    if epoch is None:
        csv_path = os.path.join(dir_path, f"{key}.csv")
    else:
        csv_path = os.path.join(dir_path, f"{key}_epoch_{epoch}.csv")
    df = pd.read_csv(csv_path)

    y_len = len(df.columns[df.columns.str.contains(key)])

    batch_idx = df['batch_idx'].tolist()
    x = df.index.tolist()
    
    # Normalize the supplemental values for colormap usage
    norm = mcolors.Normalize(vmin=min(batch_idx), vmax=max(batch_idx))
    cmap = cm.get_cmap('viridis')  # Choose a colormap


    fig, ax = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event
    title = f"{key} for epoch {epoch}" if epoch is not None else f"{key}"
    fig.suptitle(title)

    for i in range(y_len):
        y_label = f"{i}_{key}"
        y = df[y_label].tolist()
        for j in range(len(x)-1):
            ax[i].plot(x[j:j+2], y[j:j+2], color=cmap(norm(batch_idx[j])))
        ax[i].set_ylabel(y_label)
        ax[i].set_title(y_label)
        ax[i].grid(True)
        if i == y_len-1:
            ax[i].set_xlabel("frame #")

    fig.savefig(os.path.join(dir_path, f"{key}_epoch_{epoch}.png"))
    plt.close(fig)

def generate_plots(base_dir, train_or_test, epochs):
    for key in BATCH_KEYS_TO_PLOT:
        dir_path = os.path.join(base_dir, train_or_test, key)
        if epochs is not None:
            plot_per_batch(dir_path, key, [epoch for epoch in range(1, epochs+1)])
        else:
            plot_per_batch(dir_path, key, None)
        
        for key in FRAME_KEYS_TO_PLOT:
            dir_path = os.path.join(base_dir, train_or_test, key)
            if epochs is not None:
                for epoch in range(1, epochs+1):
                    plot_per_frame(dir_path, key, epoch)
            else:
                plot_per_frame(dir_path, key, None)
    
    if epochs is not None:
        for key in BATCH_KEYS_TO_PLOT:
            plot_train_vs_val(base_dir=base_dir, key=key, epochs=[epoch for epoch in range(1, epochs+1)])


if __name__ == '__main__':
    ### EDIT ME ###
    base_directory = "./results/stacked_1024_3/"
    epochs = 9
    ###############
    generate_plots(base_directory, "train", epochs)
    generate_plots(base_directory, "test", epochs)