import numpy as np
from matplotlib import pyplot as plt
import glob
import random
import time

class Vizualizer:
    def infer_and_plot(self, model, infer, title, folder_name, num_images=9, cols=3, plt_width=16, plt_height=5, out_file=None, add_border=True, title_color='white'):
        files = list(glob.glob(folder_name + '/*.jpg'))
        rows = num_images // cols
        rows = rows if rows * cols >= num_images else rows + 1

        fig, axes = plt.subplots(rows, cols, figsize=(plt_width, plt_height))
        if add_border:
            rect = plt.Rectangle(
                # (lower-left corner), width, height
                (0.00, 0.0), 1., 1., fill=False, color="k", lw=2, 
                zorder=1000, transform=fig.transFigure, figure=fig
            )
            fig.patches.extend([rect])
        axes = np.array(axes).flatten().tolist()
        times = []
        for i in range(num_images):
            ax = axes[i]
            filename = random.choice(files)
            times.append(infer(ax, model, filename))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.axis('off')
        fig.suptitle(title, fontsize=24, color=title_color)

        plt.tight_layout(pad=.05, h_pad=None, w_pad=None, rect=None)
        if out_file is not None:
            plt.savefig(out_file, bbox_inches='tight')
        plt.show()
        return times
