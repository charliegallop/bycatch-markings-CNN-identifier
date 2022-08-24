import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# define the colour palette
palette = {
    "var0": "#444c54", # charcoal grey
    "var1": "#C79B88", # Antique Brass
    "var2": "#C47454", # Copper red
    "var3": "#B6D7B9", # Turqoise green
    "var4": "#E9C46A", # maize crayola
    "var5": "#F4A261", # sandy brown
    "var6": "#998b70", # grullo
    "var7": "#fcb923", # selective yellow
    "var8": "#e24670",  # paradise pink
    "var9": "#264653" # charcoal
}

def wrangle_data(data):
    df = data

    df = df.rename(columns = {
    '               epoch':"epoch",
    '      train/box_loss':"train_b_loss",
    '      train/obj_loss':"train_ob_loss",
    '      train/cls_loss':"train_cl_loss",
    '   metrics/precision': "P",
    '      metrics/recall': "R",   
    '     metrics/mAP_0.5': "mAp_0.5",
    'metrics/mAP_0.5:0.95': "mAp_0.5:0.95",
    '        val/box_loss': "val_b_loss",    
    '        val/obj_loss': "val_ob_loss",
    '        val/cls_loss': "val_cl_loss",
    '               x/lr0': "lr0",
    '               x/lr1': "lr1",
    '               x/lr2': "lr2"
    })

    df['train_loss'] = df[ 'train_b_loss'] + df['train_ob_loss'] + df['train_cl_loss']
    df['val_loss'] = df[ 'val_b_loss'] + df['val_ob_loss'] + df['val_cl_loss']

    return df

def plot_yolo_losses(data, losses_save_as, model = None):

    if model == "yolo":
        data = wrangle_data(data)
    else:
        data = data

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20,15), sharex = True)

    
    
    x_label = "epoch"

    y_labels = [
        'train_loss',
        'train_b_loss',
        'train_ob_loss',
        'train_cl_loss',
        'val_loss',
        'val_b_loss',
        'val_ob_loss',
        'val_cl_loss',
        
    ]



    titles = [
        'Overall Loss',
        'Bounding Box Loss',
        'Objectness Loss',
        'Classifier Loss'
    ]

    it = 0
    for i in range(rows):
        for j in range(cols):
            axes[i, j].set_title(titles[j+it], fontsize = 25)

            # Training loss plot
            sns.lineplot(
                x = x_label,
                y = y_labels[j+it],
                color = palette['var1'],
                data = data,
                ax=axes[i, j],
                linewidth = 2.5
                )
           
            # Validation loss plot
            sns.lineplot(
                x = x_label,
                y = y_labels[j+4+it],
                color = palette['var0'],
                data = data,
                ax=axes[i, j],
                linewidth = 2.5
                )

            axes[i,j].legend(labels = ['Train', 'Val'])

            if j == 0:
                axes[i, j].set_ylabel("Loss", fontsize = 25)
            else:
                axes[i, j].set_ylabel("")


            axes[i, j].set_xlabel("Epoch", fontsize = 25)
        it += 2
        fig.subplots_adjust(wspace=0.15, hspace = 0.1)
        fig.savefig(losses_save_as, dpi=300, bbox_inches='tight')
                   
    return fig

def plot_yolo_metrics(data, metrics_save_as, model = None):

    if model == "yolo":
        data = wrangle_data(data)
    else:
        data = data
    rows = 2
    cols = 1
    fig, axes = plt.subplots(rows, cols, figsize=(20,8), sharex = True)
    axes[0].set_title("Metrics", fontsize = 25)
    it = 0
    x_label = 'epoch'
    
    y_labels = [
     'P',
     'R',
     'mAp_0.5',
     'mAp_0.5:0.95'
    ]

    titles = [
     'Precision',
     'Recall',
     'Mean Average Precision\nconf = 0.5',
     'Mean Average Precision\nconf = 0.5:0.95'
    ]

    titles2 = [
     'Precision',
     'Recall',
     'mAP_0.5',
     'mAP_0.5:0.95'
    ]

    for j in range(rows):
        sns.lineplot(
            x = x_label,
            y = y_labels[j+it],
            color = palette["var1"],
            data = data,
            ax=axes[j],
            linewidth = 3
        )
        sns.lineplot(
            x = x_label,
            y = y_labels[j+1+it],
            color = palette["var0"],
            data = data,
            ax=axes[j],
            linewidth = 3
        )
        
        if it == 0:
            axes[j].legend(labels = ['Precision', 'Recall'])
        else:
            axes[j].legend(labels = ['mAP_0.5', 'mAP_0.5:0.95'])
            
        axes[j].set_xlabel("Epoch", fontsize = 20)
        axes[j].set_ylabel("")
                
        it = 1
        plt.ylim(0, 1)
        fig.subplots_adjust(wspace=0.2, hspace = 0.1)
        fig.savefig(metrics_save_as, dpi=300, bbox_inches='tight')

    return fig