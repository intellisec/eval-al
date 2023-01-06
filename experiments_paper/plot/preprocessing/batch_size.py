import math
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import matplotlib as mpl

# Read result file into dataframe
def read_result_file(path, suffix) :
	df = pd.DataFrame()
	with path.open() as f :
		for line in f :
			line = [w.strip("'") for w in line.strip('\n').split(' ')]
			if line and int(line[5])>1000:
				if str(line[0].split('_')[0])=='entropy':
					line[0] = 'Entropy'
				row = pd.DataFrame({
					'Method':    str(line[0].split('_')[0]),
					'Trial':     int(line[1]),
					'Cycle':     int(line[3]),
					'N_labeled': int(line[5]),
					'Accuracy':  float(line[6]),
					'BS': suffix,
				}, index=[0])
				df = pd.concat([df, row], ignore_index=True)
	return df

def two_tailed_paired_t_test_batchsizewise(df):
	BSs = list(df['BS'].unique())
	methods = list(df['Method'].unique())
	t_values = []
	n_N = 0
	n_N_scores_dict = {}
	for BS in BSs:
		B_df = df.loc[(df['BS']==BS)]
		for m_pair in itertools.combinations(methods, 2):
			avg_b = B_df.groupby(['N_labeled'])
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
					t_value_s =  2.920 #trials 3
					if t_value > t_value_s:
						t_values.append([BS, m_pair[0], m_pair[1], N_label, t_value, True])
					elif t_value < - t_value_s:
						t_values.append([BS, m_pair[1], m_pair[0], N_label, t_value, True])
					else:
						t_values.append([BS, m_pair[0], m_pair[1], N_label, t_value, False])
						t_values.append([BS, m_pair[1], m_pair[0], N_label, t_value, False])
			n_N_scores_dict[BS]= n_N
			n_N = 0
	
	t_values_df = pd.DataFrame(t_values, columns= ['BS', 'M0', 'M1', 'n_labels', 't_value', 'score'])
	return t_values_df, n_N_scores_dict		

def plot_heatmap_individual(t_tables, means, plots_path):
	for key in t_tables:
		plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
		plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
		fig, axn = plt.subplots(2, 1, sharey=False, figsize=(20, 8), gridspec_kw={'height_ratios': [6,1]})
		cmap = sns.cm.rocket_r
		x = ['BALD', 'Badge', 'CoreSet', 'Entropy', 'Random']
		if key==1000:
		#t0
			plt.subplots_adjust(left=0.16, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.05)
			fname = f'B1000_3trials_t.png'

			h_det = sns.heatmap(ax=axn[0], data=t_tables[key], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=x, 
						cbar=None,
						cbar_kws={'format':'%.2f'},
						vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[key].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=['$\Phi$'], cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			m_h.set_yticklabels(m_h.get_yticklabels(), rotation = 180, fontsize=50)
		elif key==2000:
		#t1
			plt.subplots_adjust(left=0.105, bottom=0.05, right=0.895, top=None, wspace=None, hspace=0.05)
			fname = f'B2000_3trials_t.png'

			h_det = sns.heatmap(ax=axn[0], data=t_tables[key], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, 
						cbar=None,
						cbar_kws={'format':'%.2f'},
						vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)
			m_h = sns.heatmap(ax=axn[1], data=means[key].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
		else:
		#t2 
			plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
			bbox = axn[0].axes.get_subplotspec().get_position(fig)
			bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
			cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height+bbox1.height+0.02])
			cbar_ax.tick_params(labelsize=50)
			fname = f'B4000_3trials_t.png'

			h_det = sns.heatmap(ax=axn[0], data=t_tables[key], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, 
						cbar_ax=cbar_ax,
						cbar_kws={'format':'%.2f'},
						vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)
			m_h = sns.heatmap(ax=axn[1], data=means[key].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
		
		fpath = plots_path / fname
		print("Saved at: ", fpath)
		os.makedirs(plots_path, exist_ok=True)
		plt.savefig(fpath)

def sum_t_tables(df, n_N_scores_dict):
	BSs = list(df['BS'].unique())
	BS_tables = {}
	results = []
	for BS in BSs:
		B_df = df.loc[(df['BS'] == BS)]
		avg_config = B_df.groupby(['M0', 'M1'])
		for (m0, m1), sub_df in avg_config:
			if BS==8000:
				print(sub_df)
			N = n_N_scores_dict[BS]
			results.append([m0, m1, (sub_df['score'].sum())/N])
	res_df = pd.DataFrame(results, columns=['m1', 'm2', 'score'])
	print(res_df)
	triu_res = res_df.pivot_table(index='m1', columns='m2', values='score', fill_value=0,aggfunc= 'sum').rename_axis(None, axis=1).rename_axis(None, axis=0)
	return res_df

def t_value_metric_two_tailed_paired_t_test_batchsizewise(df, n_N_scores_dict):
	BSs = list(df['BS'].unique())
	BS_tables = {}
	means = {}
	for BS in BSs:
		B_df = df.loc[(df['BS'] == BS)]
		avg_config = B_df.groupby(['M0', 'M1'])
		results = []
		for (m0, m1), sub_df in avg_config:
			N = n_N_scores_dict[BS]
			results.append([m0+'_'+str(BS), m1+'_'+str(BS), (sub_df['score'].sum())/N])
		res_df = pd.DataFrame(results, columns=['m1', 'm2', 'score'])
		triu_res = res_df.pivot_table(index='m1', columns='m2', values='score', fill_value=0).rename_axis(None, axis=1).rename_axis(None, axis=0)
		BS_tables[BS] = triu_res
		means[BS] = triu_res.mean(axis=0).to_frame()
	return BS_tables, means

def plot_setting(task_learner, dataset, init_budget, results_path, plots_path):
	sub_dict = {1000:26, 2000:13, 4000:7}
	df = pd.DataFrame()
	for suffix in sub_dict:
		setting_str = f'{task_learner}_{dataset}_I{init_budget}_B{suffix}_c{sub_dict[suffix]}'
		# Read result files from all (method) subdirectories
		for subdir in results_path.iterdir():
			if subdir.is_dir():
				# Build result file name
				method = subdir.name
				fname = f'{method}_{setting_str}.txt'
				fpath = subdir / fname
				if fpath.exists() and method in ['Badge', 'BALD', 'entropy', 'CoreSet', 'Random']:
				    _df = read_result_file(fpath, suffix)
				    df = pd.concat([df, _df], ignore_index=True)

	# T-values
	t_values_df, n_N_scores_dict = two_tailed_paired_t_test_batchsizewise(df)
	t_tables, means = t_value_metric_two_tailed_paired_t_test_batchsizewise(t_values_df, n_N_scores_dict)
	plot_heatmap_individual(t_tables, means, plots_path)
	# t_tables = sum_t_tables(t_values_df, n_N_scores_dict)
	# fname = f't_values_cifar10im10_b2000_b4000_new.csv'
	# fpath = csv_path / fname
	# t_tables.to_csv(fpath, index=False)