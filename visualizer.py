import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

### EDIT ME ###
DIRECTORY = "./results/stacked_1024/"
TRAIN_OR_TEST = "test"
BATCH_KEYS_TO_PLOT = ["rot_l2_loss", "pos_l2_loss"]
# FRAME_KEYS_TO_PLOT = ["rot_diffs", "pos_diffs"]
FRAME_KEYS_TO_PLOT = []
# DATASET = [0,2,8,9]
DATASET = [3, 4, 5, 6, 7, 10]
EPOCHS = [1,2,3,4,5,6,7,8,9,10]
###############

# Function to close plot when "Esc" is pressed
def _on_key(event):
    if event.key == "escape":
        plt.close('all')

# Plots L2 losses in a dataset, with each point being the l2 loss for the batch.
# Plots multiple lines for each epoch.
def plot_per_batch(dir_path, key, dataset, epochs=None):
    dataset = f"0{dataset}" if dataset < 10 else str(dataset)

    if epochs is None:
        csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}.csv")
        dfs = pd.read_csv(csv_path)
    else:
        dfs = []
        for epoch in epochs:
            csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}_epoch_{epoch}.csv")
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

    title = f"{key} for dataset {dataset}"
    fig.suptitle(title)
    ax.set_xlabel("batch #")
    ax.set_ylabel(key)
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(dir_path, f"{key}_dataset_{dataset}.png"))
    plt.close(fig)

# Only generates plot for a single epoch.
def plot_per_frame(dir_path, key, dataset, epoch=None):
    dataset = f"0{dataset}" if dataset < 10 else str(dataset)

    if epoch is None:
        csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}.csv")
    else:
        csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}_epoch_{epoch}.csv")
    df = pd.read_csv(csv_path)

    y_len = len(df.columns[df.columns.str.contains(key)])

    batch_idx = df['batch_idx'].tolist()
    x = df.index.tolist()
    
    # Normalize the supplemental values for colormap usage
    norm = mcolors.Normalize(vmin=min(batch_idx), vmax=max(batch_idx))
    cmap = cm.get_cmap('viridis')  # Choose a colormap


    fig, ax = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event
    title = f"{key} for dataset {dataset}: epoch {epoch}" if epoch is not None else f"{key} for dataset {dataset}"
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

    fig.savefig(os.path.join(dir_path, f"{key}_dataset_{dataset}_epoch_{epoch}.png"))
    plt.close(fig)

if __name__ == '__main__':
    base_path = os.path.join(DIRECTORY, TRAIN_OR_TEST)
    for key in BATCH_KEYS_TO_PLOT:
        for dataset in DATASET:
            dir_path = os.path.join(base_path, key)
            plot_per_batch(dir_path, key, dataset, EPOCHS)
                
    for key in FRAME_KEYS_TO_PLOT:
        for dataset in DATASET:
            if EPOCHS is not None:
                dir_path = os.path.join(base_path, key)
                for epoch in EPOCHS:
                    plot_per_frame(dir_path, key, dataset, epoch)
            else:
                plot_per_frame(dir_path, key, dataset, None)