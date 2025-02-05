import pandas as pd
import matplotlib.pyplot as plt


CSV_NAME = "./results/test/default_epoch_0.csv"

# Function to close plot when "Esc" is pressed
def _on_key(event):
    if event.key == "escape":
        plt.close()

def plot_l2_losses_per_dataset():
    df = pd.read_csv(CSV_NAME)

    # Get number of datasets
    dataset_indexes = sorted(df['dataset_idx'].unique())

    l2_losses = dict()
    for idx in dataset_indexes:
        l2_losses[idx] = df[df['dataset_idx'] == idx].reset_index(drop=True)['l2_loss']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event
    for dataset_idx, val in l2_losses.items():
        plt.plot(val.index, val, label=f'dataset_{dataset_idx}')

    plt.xlabel("batch #")
    plt.ylabel("l2_loss")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_l2_losses_per_batch():
    df = pd.read_csv(CSV_NAME)['l2_loss']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect("key_press_event", _on_key)  # Bind the key event

    plt.plot(df.index, df)

    plt.xlabel("batch #")
    plt.ylabel("l2_loss")
    plt.grid(True)

    plt.show()




if __name__ == '__main__':
    # plot_l2_losses_per_dataset()
    plot_l2_losses_per_batch()