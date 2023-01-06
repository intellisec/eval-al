import math
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os

from scipy.special import stdtrit

# Read result file into dataframe
def read_result_file(path) :
	df = pd.DataFrame()
	with path.open() as f :
		for line in f :
			line = [w.strip("'") for w in line.strip('\n').split(' ')]
			if line and int(line[5])>=1001:
				if int(line[1]) <=3:
					row = pd.DataFrame({
						'Method':    line[0].split('_')[0],
						'Trial':     int(line[1]),
						'Cycle':     int(line[3]),
						'N_labeled': int(line[5]),
						'Accuracy':  float(line[6]),
						'Type': 'tinyimagenet',
					}, index=[0])
					df = pd.concat([df, row], ignore_index=True)
	return df

def t_value_metric_two_tailed_paired_t_test(df, n_N_scores_dict):
	tables = {}
	means = {}
	type_df = df.groupby(['Type'])
	for type, sub_df in type_df:
		avg_config = sub_df.groupby(['M0', 'M1'])
		results = []
		for (m0, m1), sub_df in avg_config:
			N = n_N_scores_dict[type]
			results.append([m0, m1, (sub_df['score'].sum())/N])
		res_df = pd.DataFrame(results, columns=['m1', 'm2', 'score'])
		triu_res = res_df.pivot_table(index='m1', columns='m2', values='score', fill_value=0).rename_axis(None, axis=1).rename_axis(None, axis=0)
		tables[type] = triu_res
		means[type] = triu_res.mean(axis=0).to_frame()
	return tables, means


def two_tailed_paired_t_test(df):
	methods = list(df['Method'].unique())
	t_values = []
	n_N_scores_dict = {}
	group_df = df.groupby(['Type'])
	for type, type_df in group_df:
		for m_pair in itertools.combinations(methods, 2):
			avg_b = type_df.groupby(['N_labeled'])
			n_N = 0
			for (N_label), sub_df in avg_b:
				sub_df = sub_df.loc[(sub_df['Method'] == m_pair[0]) | (sub_df['Method'] == m_pair[1])]
				if len(list(sub_df['Method'].unique())) == 2:
					n_N += 1
					sub_df_g = sub_df.groupby(['Trial'])
					acc_diff = []
					for _, sub_sub_df in sub_df_g:
						acc_diff.append(sub_sub_df.loc[sub_df['Method'] == m_pair[0]]['Accuracy'].values[0] -
										sub_sub_df.loc[sub_df['Method'] == m_pair[1]]['Accuracy'].values[0])
					mean_difference = np.array(acc_diff).mean()
					std = np.array(acc_diff).std()
					n = len(acc_diff)
					std_error = std / math.sqrt(n)
					t_value = mean_difference / std_error
					t_value_2dof_90conf_twotailed =  2.920
					if t_value > t_value_2dof_90conf_twotailed:
						t_values.append([type, m_pair[0], m_pair[1], N_label, t_value, True])
					elif t_value < - t_value_2dof_90conf_twotailed:
						t_values.append([type, m_pair[1], m_pair[0], N_label, t_value, True])
					else:
						t_values.append([type, m_pair[0], m_pair[1], N_label, t_value, False])
						t_values.append([type, m_pair[1], m_pair[0], N_label, t_value, False])
		n_N_scores_dict[type]= n_N
	t_values_df = pd.DataFrame(t_values, columns= ['Type', 'M0', 'M1', 'n_labels', 't_value', 'score'])
	return t_values_df, n_N_scores_dict

def plot_heatmap_individual(task_learner, dataset, init_budget, df_dict, means, plots_path):
	for key in df_dict:
		plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
		plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
		fig, axn = plt.subplots(2, 1, sharey=False, figsize=(17, 7), gridspec_kw={'height_ratios': [5,1]})
		x = ['BALD', 'CoreSet','Entropy', 'Random']
		
		plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
		bbox = axn[0].axes.get_subplotspec().get_position(fig)
		bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
		cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height+bbox1.height+0.02])
		cbar_ax.tick_params(labelsize=50)
	
		cmap = sns.cm.rocket_r
		columns_titles = ["BALD", "CoreSet", "entropy","Random"]
		rows_titles = ["BALD", "CoreSet", "entropy","Random"]
		df_dict['tinyimagenet']=df_dict['tinyimagenet'].reindex(columns=columns_titles)
		df_dict['tinyimagenet']=df_dict['tinyimagenet'].reindex(index=columns_titles)
		x = ['BALD', 'CoreSet','Entropy', 'Random']
		h_det = sns.heatmap(ax=axn[0], data=df_dict['tinyimagenet'], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, cbar_ax=cbar_ax, vmin=0, vmax=1, cbar_kws={'format':'%.2f'}, annot_kws={"fontsize": 50}, fmt='.2f')
		axn[0].tick_params(axis='both', which='major', pad=9)
		# h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
		h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)
		
		m_h = sns.heatmap(ax=axn[1], data=means['tinyimagenet'].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
		axn[1].tick_params(axis='both', which='major', pad=20)
		fname = f'tinyimagenet_h_t.pdf'

		fpath = plots_path / fname
		os.makedirs(plots_path, exist_ok=True)
		plt.savefig(fpath)



def plot_setting(task_learner, dataset, init_budget, budget, cycles, results_path, plots_path):
	df = pd.DataFrame()
	for fpath in results_path.glob('**/*.txt'):
		# Append extracted results to dataframe
		if 'tinyimagenet' in str(fpath):
			_df = read_result_file(fpath)
			df = pd.concat([df, _df])
	df = df.reset_index(drop=True)

	t_values_df, n_N_scores_dict = two_tailed_paired_t_test(df)
	t_tables, means = t_value_metric_two_tailed_paired_t_test(t_values_df, n_N_scores_dict)
	plot_heatmap_individual(task_learner, dataset, init_budget, t_tables, means, plots_path)
	return

