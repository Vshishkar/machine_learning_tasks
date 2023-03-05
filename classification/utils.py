import matplotlib.pyplot as plt
import os
import numpy as np

def plot_data(data_dir):
    rows = 17
    cols = 6
    fig, ax = plt.subplots(rows, cols, figsize=(25, 25))
    fig.suptitle("Showing random picture from each class", y=1.05, fontsize=22)
    foods_sorted = sorted(os.listdir(data_dir))
    food_id = 0
    for i in range(rows):
        for j in range(cols):
            try:
                food_selected = foods_sorted[food_id]
                food_id += 1
            except:
                break
            if food_selected == ".DS_Store":
                continue
            food_selected_images = os.listdir(os.path.join(data_dir, food_selected))
            food_selected_random = np.random.choice(food_selected_images)
            image = plt.imread(os.path.join(data_dir, food_selected, food_selected_random))
            ax[i][j].imshow(image)
            ax[i][j].set_title(food_selected, pad=5)

    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()