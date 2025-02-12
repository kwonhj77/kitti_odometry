import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

DIRECTORY = "./results/quick_stacked/"
TRAIN_OR_TEST = "train"
BATCH_KEYS_TO_PLOT = ["rot_l2_loss", "pos_l2_loss"]
FRAME_KEYS_TO_PLOT = ["rot_diffs", "pos_diffs"]
DATASET = [3,]
EPOCHS = [1,]

# Function to close plot when "Esc" is pressed
def _on_key(event):
    if event.key == "escape":
        plt.close()

# Plots L2 losses in a dataset, with each point being the l2 loss for the batch.
def plot_per_batch(dir_path, key, dataset, epoch=None):
    dataset = f"0{dataset}" if dataset < 10 else str(dataset)

    if epoch is None:
        csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}.csv")
    else:
        csv_path = os.path.join(dir_path, f"{key}_dataset_{dataset}_epoch_{epoch}.csv")
    df = pd.read_csv(csv_path)

    batch_idx = df['batch_idx']
    y = df[key]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event
    title = f"{key} for dataset {dataset}: epoch {epoch}" if epoch is not None else f"{key} for dataset {dataset}"
    fig.suptitle(title)
    ax.plot(batch_idx, y)
    ax.set_xlabel("batch #")
    ax.set_ylabel(key)
    ax.grid(True)

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


    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event
    title = f"{key} for dataset {dataset}: epoch {epoch}" if epoch is not None else f"{key} for dataset {dataset}"
    fig.suptitle(title)

    for i in range(y_len):
        y = df[f"{i}_{key}"].tolist()
        for j in range(len(x)):
            ax[i].plot(x[j], y[j], color=cmap(norm(batch_idx[j])))
        ax[i].set_xlabel("frame #")
        ax[i].set_ylabel(y)
        ax[i].set_title(y)
        ax[i].grid(True)




if __name__ == '__main__':
    base_path = os.path.join(DIRECTORY, TRAIN_OR_TEST)
    for key in BATCH_KEYS_TO_PLOT:
        for dataset in DATASET:
            if EPOCHS is not None:
                for epoch in EPOCHS:
                    dir_path = os.path.join(base_path, key)
                    plot_per_batch(dir_path, key, dataset, epoch)
            else:
                pass
    for key in FRAME_KEYS_TO_PLOT:
        for dataset in DATASET:
            if EPOCHS is not None:
                for epoch in EPOCHS:
                    dir_path = os.path.join(base_path, key)
                    plot_per_frame(dir_path, key, dataset, epoch)
            else:
                pass
    plt.show()

    # plot_l2_losses_per_dataset()
    # plot_l2_losses_per_batch()