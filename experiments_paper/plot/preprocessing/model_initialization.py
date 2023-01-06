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
				if str(line[0])=='entropy':
					line[0] = 'Entropy'
				row = pd.DataFrame({
					'Method':    line[0],
					'Trial':     int(line[1]),
					'Cycle':     int(line[3]),
					'N_labeled': int(line[5]),
					'Accuracy':  float(line[6]),
					'Modelseed': suffix,
				}, index=[0])
				df = pd.concat([df, row], ignore_index=True)
	return df

def two_tailed_paired_t_test_initsetseedwise(df):
	MSs = list(df['Modelseed'].unique())
	ISs = list(df['Trial'].unique()) 
	methods = list(df['Method'].unique())
	t_values = []
	n_N = 0
	n_N_scores_dict = {}
	for IS in ISs:
		I_df = df.loc[(df['Trial']==IS)]
		for m_pair in itertools.combinations(methods, 2):
			avg_b = I_df.groupby(['N_labeled'])
			for (N_label), sub_df in avg_b:
				sub_df = sub_df.loc[(sub_df['Method'] == m_pair[0]) | (sub_df['Method'] == m_pair[1])]
				if len(list(sub_df['Method'].unique())) == 2:
					n_N += 1
					sub_df_g = sub_df.groupby(['Modelseed']) 
					acc_diff = []
					for _, sub_sub_df in sub_df_g:
						
						acc_diff.append(sub_sub_df.loc[sub_df['Method'] == m_pair[0]]['Accuracy'].values[0] - 
										sub_sub_df.loc[sub_df['Method'] == m_pair[1]]['Accuracy'].values[0])
				
					mean_difference = np.array(acc_diff).mean()
					std = np.array(acc_diff).std()
					n = len(acc_diff)		
					std_error = std / math.sqrt(n)
					t_value = mean_difference / std_error
					t_value_s =  2.920 
					if t_value > t_value_s:
						t_values.append([IS, m_pair[0], m_pair[1], N_label, t_value, True])
					elif t_value < - t_value_s:
						t_values.append([IS, m_pair[1], m_pair[0], N_label, t_value, True])
					else:
						t_values.append([IS, m_pair[0], m_pair[1], N_label, t_value, False])
						t_values.append([IS, m_pair[1], m_pair[0], N_label, t_value, False])
			n_N_scores_dict[IS]= n_N
			n_N = 0
	t_values_df = pd.DataFrame(t_values, columns= ['IS', 'M0', 'M1', 'n_labels', 't_value', 'score'])
	return t_values_df, n_N_scores_dict		

def t_value_metric_two_tailed_paired_t_test_initsetseedwise(df, n_N_scores_dict):
	ISs = list(df['IS'].unique())
	IS_tables = {}
	IS_means = {}
	for IS in ISs:
		I_df = df.loc[(df['IS'] == IS)]
		avg_config = I_df.groupby(['M0', 'M1'])
		results = []
		for (m0, m1), sub_df in avg_config:
			N = n_N_scores_dict[IS]
			results.append([m0+'_'+str(IS), m1+'_'+str(IS), (sub_df['score'].sum())/N])
		res_df = pd.DataFrame(results, columns=['m1', 'm2', 'score'])
		triu_res = res_df.pivot_table(index='m1', columns='m2', values='score', fill_value=0).rename_axis(None, axis=1).rename_axis(None, axis=0)	
		IS_tables[IS] = triu_res
		IS_means[IS]=triu_res.mean(axis=0).to_frame()
	return IS_tables, IS_means
def plot_heatmap_individual(t_tables, means, plots_path):
	for key in t_tables:
		plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
		plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
		fig, axn = plt.subplots(2, 1, sharey=False, figsize=(20, 8), gridspec_kw={'height_ratios': [6,1]})
		cmap = sns.cm.rocket_r
		x = ['BALD', 'Badge', 'CoreSet', 'Entropy', 'Random']
		if key==1:
		#t0
			plt.subplots_adjust(left=0.16, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.05)
			h_det = sns.heatmap(ax=axn[0], data=t_tables[1], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=x, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f') #cbar_ax=cbar_ax
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[1].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=['$\Phi$'], cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			m_h.set_yticklabels(m_h.get_yticklabels(), rotation = 180, fontsize=50)
			fname = f'modelinitseed0_3trials_t.png'
		#t1
		elif key==2:
			plt.subplots_adjust(left=0.105, bottom=0.05, right=0.895, top=None, wspace=None, hspace=0.05)
			h_det = sns.heatmap(ax=axn[0], data=t_tables[2], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f') #cbar_ax=cbar_ax
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[2].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
							yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			fname = f'modelinitseed1_3trials_t.png'
		#t2
		else:
			plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
			bbox = axn[0].axes.get_subplotspec().get_position(fig)
			bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
			cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height+bbox1.height+0.02])
			cbar_ax.tick_params(labelsize=50)
			h_det = sns.heatmap(ax=axn[0], data=t_tables[3], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, cbar_ax=cbar_ax, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f') #cbar_ax=cbar_ax
			axn[0].tick_params(axis='both', which='major', pad=18)
			h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[3].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
						yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			fname = f'modelinitseed2_3trials_t.png'
	
		
		fpath = plots_path / fname
		print("Saved at: ", fpath)
		os.makedirs(plots_path, exist_ok=True)
		plt.savefig(fpath)

def plot_setting(task_learner, dataset, init_budget, budget, cycles, results_path, plots_path):
	sub_dict = {'modelseed0', 'modelseed1', 'modelseed2'} 
	df = pd.DataFrame()
	for suffix in sub_dict:
		setting_str = f'{task_learner}_{dataset}_I{init_budget}_B{budget}_c{cycles}_{suffix}'
		# Read result files from all (method) subdirectories
		for subdir in results_path.iterdir():
			if subdir.is_dir():
				# Build result file name
				method = subdir.name
				fname = f'{method}_{setting_str}.txt'
				fpath = subdir / fname
				if fpath.exists():
					_df = read_result_file(fpath, suffix)
					df = pd.concat([df, _df], ignore_index=True)

	## plot initsetseed -wise heatmap
	# T-value
	t_values_df, n_N_scores_dict = two_tailed_paired_t_test_initsetseedwise(df)
	t_tables, means = t_value_metric_two_tailed_paired_t_test_initsetseedwise(t_values_df, n_N_scores_dict)
	plot_heatmap_individual(t_tables, means, plots_path)
