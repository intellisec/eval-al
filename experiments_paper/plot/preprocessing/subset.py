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

results_path = Path('../results_paper/early_stop/') #3trial
# results_path = Path('../results_paper/results_final/subset/') # 5trial
csv_path = Path('../results/early_stop/t-values')
plots_path = Path('../plots/early_stop/')
os.makedirs(plots_path, exist_ok=True)
os.makedirs(csv_path, exist_ok=True)
alpha = 0.05

# Argument parser for command line use
parser = ArgumentParser(description = 'Plot and compare AL strategies for a given training setting')
parser.add_argument('-t', '--task_learner', type = str, required = True,  help = 'Name of task learning model')
parser.add_argument('-d', '--dataset',      type = str, required = True,  help = 'Learning dataset')
parser.add_argument('-i', '--init_budget',  type = int, required = True,  help = 'Initial labeled pool size')
parser.add_argument('-b', '--budget',       type = int, required = False,  help = 'Query budget per cycle')
parser.add_argument('-c', '--cycles',       type = int, required = False,  help = 'Number of query iterations')
parser.add_argument('-s', '--sub_pre',      type = int, required = False, help = 'Subset size (optional)')


def Calu_avg_on_initset(df):
	avg = pd.DataFrame()
	grouped = df.groupby(['Method', 'N_labeled', 'Model_init'])
	for (method, N_labeled, init_model), sub_df in grouped:
		row = pd.DataFrame({
			'Method' : method,
			'N_labeled' : int(N_labeled),
			'Mean_acc' : float(sub_df['Accuracy'].mean()),
			'Std_acc' : float(sub_df['Accuracy'].std()),
			'init model' : init_model
		}, index=[0])
		avg = pd.concat([avg, row], ignore_index=True)
	return avg

# Read result file into dataframe
def read_result_file(path,typ) :
	df = pd.DataFrame()
	with path.open() as f :
		for line in f :
			line = [w.strip("'") for w in line.strip('\n').split(' ')]
			if line and int(line[5])>1000:
				row = pd.DataFrame({
					'Method':    line[0].split('_')[0],
					'Trial':     int(line[1]),
					'Cycle':     int(line[3]),
					'N_labeled': int(line[5]),
					'Accuracy':  float(line[6]),
					'Type': typ,
				}, index=[0])
				df = pd.concat([df, row], ignore_index=True)
	return df

def get_t_table_value(std_Det, std_ND, n_Det, n_ND):
	std_error_Det = std_Det ** 2 / n_Det
	std_error_ND = std_ND ** 2 / n_ND

	nominator = (std_error_Det + std_error_ND) ** 2
	denominator = std_Det ** 4 / (n_Det ** 2 * (n_Det - 1)) + std_ND ** 4 / (n_ND ** 2 * (n_ND - 1))
	dof = nominator/denominator
	#dof = round((std_error_Det + std_err_2)**2 / ((std_err_1/(n_1-1)) + (std_err_2/(n_2-1))))
	alpha = 0.05
	return stdtrit(dof, 1 - alpha)

# Computes the t-test for each nondeterministic trial
# FullsetT1 - SubSetT1
# FullsetT1 - SubSetT2
# FullsetT1 - SubSetT3
# etc.
def paired_t_test(df):
	# Compute the differences between subset and non-subset accuracies
	avg_method = df.groupby(['Method', 'N_labeled', 'Trial'])
	t_df = pd.DataFrame()
	for (method, N_label, trial), sub_df in avg_method:
		row = pd.DataFrame({
			'Method': method,
			'N_labeled': N_label,
			'Acc_difference': sub_df.loc[sub_df['Model_init'] == 'Fullset']['Accuracy'].values[0] - sub_df.loc[sub_df['Model_init'] == 'Subset']['Accuracy'].values[0],
		}, index=[0])
		t_df = pd.concat([t_df, row], ignore_index=True)

	# Calculate the t-value for each method. They are negative which indicates that the nondeterministic mean is larger than the deterministic mean
	t_method = t_df.groupby(['Method', 'N_labeled'])
	t_values = []
	for (method, n_label), sub_df in t_method:
		mean_difference = sub_df['Acc_difference'].mean()
		std = sub_df['Acc_difference'].std()
		n = sub_df.shape[0]
		t_table_value_95conf = stdtrit(n - 1, 1-alpha)
		std_error = std / math.sqrt(n)
		t_value = mean_difference / std_error
		t_values.append([method, method, n_label, abs(t_value), abs(t_value)<t_table_value_95conf])
	t_values_df = pd.DataFrame(t_values, columns=['Method0', 'Method1', 'n_labels', 't_value', 'accept_null_hypothesis'])
	return t_values_df



def welch_test(df):
	subset_df = df.loc[(df['Model_init'] == 'Subset')]
	fullset_df = df.loc[(df['Model_init'] != 'Fullset')]

	subset_det_each_N_df = pd.DataFrame()
	grouped = subset_df.groupby(['Method', 'N_labeled'])
	
	for (method, N_labeled),sub_df in grouped:
	
		row = pd.DataFrame(
			{
				'Method': method,
				'N_labeled' : int(N_labeled),
				'Mean_Acc': float(sub_df['Accuracy'].mean()),
				'Std_Acc': float(sub_df['Accuracy'].std()),
				'Model_init' : 'Subset'
			},index=[0]
		)
		subset_det_each_N_df = pd.concat([subset_det_each_N_df, row], ignore_index=True)

	fullset_each_N_df = pd.DataFrame()
	grouped = fullset_df.groupby(['Method', 'N_labeled'])
	
	for (method, N_labeled), sub_df in grouped:
	
		row = pd.DataFrame(
			{
				'Method': method,
				'N_labeled': int(N_labeled),
				'Mean_Acc': float(sub_df['Accuracy'].mean()),
				'Std_Acc': float(sub_df['Accuracy'].std()),
				'Model_init' : 'Fullset'
			}, index=[0]
		)
		fullset_each_N_df = pd.concat([fullset_each_N_df, row], ignore_index=True)

	t_df = pd.concat([fullset_each_N_df, subset_det_each_N_df], ignore_index=True)

	t_method = t_df.groupby(['Method', 'N_labeled'])
	t_values = []
	for (method, n_label), sub_df in t_method:
	
		mean_difference = sub_df.loc[sub_df['Model_init'] == 'Fullset']['Mean_Acc'].values[0] - sub_df.loc[sub_df['Model_init'] == 'Subset']['Mean_Acc'].values[0]
		n = 3
		std_fullset = sub_df.loc[sub_df['Model_init'] == 'Fullset']['Std_Acc'].values[0]
		std_subset = sub_df.loc[sub_df['Model_init'] == 'Subset']['Std_Acc'].values[0]
		std_error = math.sqrt(std_fullset**2/n + std_subset**2/n)
		t_table_value = get_t_table_value(std_fullset, std_subset, n, n)
	
		t_value = mean_difference / std_error
	
		t_values.append([method, method, n_label, abs(t_value), abs(t_value)<t_table_value])
	t_values_df = pd.DataFrame(t_values, columns=['Method0', 'Method1', 'n_labels', 't_value', 'accept_null_hypothesis'])
	return t_values_df

def t_test_strategies_allconfigs(df):
	methods = list(df['Method'].unique())
	# calculate pair wise comparision
	t_values = []
	for m_pair in itertools.combinations(methods, 2):
		avg_method = df.groupby(['N_labeled'])
		for (N_label), sub_df in avg_method:
			sub_df = sub_df.loc[(sub_df['Method'] == m_pair[0]) | (sub_df['Method'] == m_pair[1])]
			sub_df_g = sub_df.groupby(['Trial', 'Model_init'])
			
			acc_diff = []
			for _, sub_sub_df in sub_df_g:
				acc_diff.append(sub_sub_df.loc[sub_df['Method'] == m_pair[0]]['Accuracy'].values[0] -
									  sub_sub_df.loc[sub_df['Method'] == m_pair[1]]['Accuracy'].values[0])

			mean_difference = np.array(acc_diff).mean()
			std = np.array(acc_diff).std()
			n = len(acc_diff)
		
			std_error = std / math.sqrt(n)
			t_value = mean_difference / std_error
			t_table_value_95conf = stdtrit(n-1, 1-alpha)
			t_values.append([m_pair[0], m_pair[1], N_label, abs(t_value), abs(t_value)<t_table_value_95conf])
			t_values.append([m_pair[1], m_pair[0], N_label, abs(t_value), abs(t_value)<t_table_value_95conf])
	t_values_df = pd.DataFrame(t_values, columns=['Method0', 'Method1', 'n_labels', 't_value', 'accept_null_hypothesis'])
	return t_values_df

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
					t_value_s =  2.920 
					if t_value > t_value_s:
						t_values.append([type, m_pair[0], m_pair[1], N_label, t_value, True])
					elif t_value < - t_value_s:
						t_values.append([type, m_pair[1], m_pair[0], N_label, t_value, True])
					else:
						t_values.append([type, m_pair[0], m_pair[1], N_label, t_value, False])
						t_values.append([type, m_pair[1], m_pair[0], N_label, t_value, False])
						
		n_N_scores_dict[type]= n_N
	t_values_df = pd.DataFrame(t_values, columns= ['Type', 'M0', 'M1', 'n_labels', 't_value', 'score'])
	return t_values_df, n_N_scores_dict

def t_value_metric(df):
	avg_method = df.groupby(['Method0','Method1'])
	results = []
	for (method0, method1), sub_df in avg_method:
		sub_df['accept_null_hypothesis'].sum()
		results.append([method0, method1, (1-sub_df['accept_null_hypothesis']).sum()])
		if method0 != method1:
			results.append([method1, method0, (1-sub_df['accept_null_hypothesis']).sum()])
	res_df = pd.DataFrame(results, columns=['m1', 'm2', 'val'])
	triu_res = res_df.pivot_table(index='m1', columns='m2', values='val')
	return triu_res

def plot_heatmap_individual(task_learner, dataset, init_budget, df_dict, means, plots_path):
	for key in df_dict:
		plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
		plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
		fig, axn = plt.subplots(2, 1, sharey=False, figsize=(19, 7), gridspec_kw={'height_ratios': [5,1]})
		cmap = sns.cm.rocket_r
		x = ['BALD', 'Badge', 'CoreSet', 'Entropy', 'Random']
		#fullset
		if key=='ES':
			columns_titles = ["BALD", "Badge","CoreSet", "entropy","Random"]
			rows_titles = ["BALD", "Badge","CoreSet", "entropy","Random"]
			df_dict['ES']=df_dict['ES'].reindex(columns=columns_titles)
			df_dict['ES']=df_dict['ES'].reindex(index=columns_titles)
			means['ES']=means['ES'].reindex(index=columns_titles)
			plt.subplots_adjust(left=0.18, bottom=0.05, right=0.97, top=None, wspace=None, hspace=0.05)
			h_det = sns.heatmap(ax=axn[0], data=df_dict[key], robust=True, annot=True, cmap=cmap, xticklabels='auto',
						yticklabels='auto', cbar=None, cbar_kws={'format':'%.2f'}, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[0].tick_params(axis='both', which='major', pad=9)
			h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[key].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
							yticklabels=['$\Phi$'], cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			m_h.set_yticklabels(m_h.get_yticklabels(), rotation = 180, fontsize=50)
			fname = f'ES_3trials_t.png'
		# subset
		else:
			columns_titles = ["BALD", "Badge","CoreSet", "entropy","Random"]
			rows_titles = ["BALD", "Badge","CoreSet", "entropy","Random"]
			df_dict['NES']=df_dict['NES'].reindex(columns=columns_titles)
			df_dict['NES']=df_dict['NES'].reindex(index=columns_titles)
			means['NES']=means['NES'].reindex(index=columns_titles)
			plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
			bbox = axn[0].axes.get_subplotspec().get_position(fig)
			bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
			cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height+bbox1.height+0.02])
			cbar_ax.tick_params(labelsize=50)

			h_det = sns.heatmap(ax=axn[0], data=df_dict[key], robust=True, annot=True, cmap=cmap, xticklabels=x,
						yticklabels=False, cbar_ax=cbar_ax, cbar_kws={'format':'%.2f'}, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[0].tick_params(axis='both', which='major', pad=9)
			# h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
			h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

			m_h = sns.heatmap(ax=axn[1], data=means[key].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
							yticklabels=False, cbar = None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
			axn[1].tick_params(axis='both', which='major', pad=20)
			fname = f'NES_3trials_t.png'
		# Deterministic
		
		fpath = plots_path / fname
		print("Saved at: ", fpath)
		os.makedirs(plots_path, exist_ok=True)
		plt.savefig(fpath)

def plot_heatmap(task_learner, dataset, init_budget, df_dict):
	fig, axn = plt.subplots(1, 2, sharey=False, figsize=(43, 15))

	bbox = axn[0].axes.get_subplotspec().get_position(fig)
	cbar_ax = fig.add_axes([0.93, bbox.y0, 0.03, bbox.height])
	plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
	cbar_ax.tick_params(labelsize=50)
	# Deterministic
	x = ['BALD', 'Badge', 'CoreSet', 'Entropy']
	h_det = sns.heatmap(ax=axn[0], data=df_dict['Fullset'], robust=True, annot=True, cmap='viridis', xticklabels=x,
					yticklabels=x, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 55})
	axn[0].tick_params(axis='both', which='major', pad=20)
	h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=45, fontweight='bold')
	h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=45, fontweight='bold')
	
	h_nd = sns.heatmap(ax=axn[1], data=df_dict['Subset'], robust=True, annot=True, cmap='viridis', xticklabels=x,
					yticklabels=False, cbar_ax=cbar_ax, vmin=0, vmax=1, annot_kws={"fontsize": 55})
	
	axn[1].tick_params(axis='x', which='major', pad=20)
	h_nd.set_xticklabels(h_nd.get_xticklabels(), rotation = 0, fontsize=45, fontweight='bold')


	fname = f'sub_heatmap_0710.pdf'
	fpath = plots_path / fname
	print("Saved at: ", fpath)
	os.makedirs(plots_path, exist_ok=True)
	plt.savefig(fpath)


def plot_setting(task_learner, dataset, init_budget, budget, cycles, sub_pre=None):
	df = pd.DataFrame()
	for fpath in results_path.glob('**/*.txt'):
		if 'e250_es' in str(fpath):
			typ='ES'
			_df = read_result_file(fpath, typ)
			df = pd.concat([df, _df])
		if 'e250.txt' in str(fpath):
			typ='NES'
			_df = read_result_file(fpath, typ)
			df = pd.concat([df, _df])
	df = df.reset_index(drop=True)

	# T-values
	t_values_df, n_N_scores_dict = two_tailed_paired_t_test(df)
	t_tables, means = t_value_metric_two_tailed_paired_t_test(t_values_df, n_N_scores_dict)
	plot_heatmap_individual(task_learner, dataset, init_budget, t_tables, means, plots_path)
	return

if __name__ == '__main__' :
	args = parser.parse_args()
	args = vars(args)
	plot_setting(**args)
