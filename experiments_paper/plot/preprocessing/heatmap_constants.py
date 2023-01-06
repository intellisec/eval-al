import matplotlib.text as mtext
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.patches as patches
import seaborn as sns

# Argument parser for command line use
parser = ArgumentParser(description='Plot and compare AL strategies for a given training setting')
parser.add_argument('-t', '--task_learner', type=str, required=True, help='Name of task learning model')
parser.add_argument('-d', '--dataset',      type=str, required=False, help='Learning dataset')
parser.add_argument('-i', '--init_budget',  type=int, required=True, help='Initial labeled pool size')

def make_heatmap(ax, df, cbar_ax):
	return sns.heatmap(ax=ax, data=df, robust=True, annot=True, cmap='viridis', xticklabels='auto',
					yticklabels='auto', cbar_ax=cbar_ax, vmin=0, vmax=1)#, annot_kws={"weight": "bold"})

def add_cbar_ax(ax, fig, left=0.93, width=0.03):
	bbox = ax.axes.get_subplotspec().get_position(fig)
	return fig.add_axes([left, bbox.y0, width, bbox.height])

def adjust_subplots(plt, left=0.09, bottom=None, right=None, top=None, wspace=0.2, hspace=None):
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

def set_ax_title(ax, title):
	ax.set_title(title)#, fontweight='bold')#, fontsize=15)

def set_label_style(ax):
	ax.set_xlabel(ax.get_xlabel(), labelpad = 1000)
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		...
		#label.set_fontweight('bold')
		#label.set_fontsize(12)

def y_label_rotation(ax):
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig_size_width = 16
fig_size_height = 16
fig_size_single = (fig_size_width, fig_size_height)
fig_size_double = (fig_size_width*2, fig_size_height)


