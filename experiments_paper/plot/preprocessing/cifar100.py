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

results_path = Path('../results_paper/results_final/batch_size/')

plots_path = Path('../plots_final/cifar100/')
os.makedirs(plots_path, exist_ok=True)
# os.makedirs(csv_path, exist_ok=True)
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
						'Type': 'cifar100',
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
					# print(sub_df)
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

def plot_heatmap_individual(task_learner, dataset, init_budget, df_dict, means):
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	fig, axn = plt.subplots(2, 1, sharey=False, figsize=(17, 7), gridspec_kw={'height_ratios': [5,1]})
	# fig, axn = plt.subplots(1, 1, sharey=False, figsize=(17, 6))

	#cifar100
	plt.subplots_adjust(left=0.18, bottom=0.05, right=0.97, top=None, wspace=None, hspace=0.05)
	
	# tinyimagenet
	# plt.subplots_adjust(left=0.06, bottom=0.05, right=0.85, top=None, wspace=None, hspace=0.05)
	# bbox = axn[0].axes.get_subplotspec().get_position(fig)
	# bbox1 = axn[1].axes.get_subplotspec().get_position(fig)
	# bbox = axn.axes.get_subplotspec().get_position(fig)
	# cbar_ax = fig.add_axes([0.88, 0.05, 0.02, bbox.height+bbox1.height+0.02])
	# cbar_ax.tick_params(labelsize=50)
	# plt.subplots_adjust(left=0.06, bottom=0.17, right=0.85, top=None, wspace=None, hspace=None)
	# bbox = axn.axes.get_subplotspec().get_position(fig)
	# cbar_ax = fig.add_axes([0.88, 0.17, 0.02, bbox.height])
	# cbar_ax.tick_params(labelsize=50)
	# Deterministic
	cmap = sns.cm.rocket_r
	#switch the column and row
	columns_titles = ["BALD", "CoreSet", "entropy","Random"]
	rows_titles = ["BALD", "CoreSet", "entropy","Random"]
	df_dict['cifar100']=df_dict['cifar100'].reindex(columns=columns_titles)
	df_dict['cifar100']=df_dict['cifar100'].reindex(index=columns_titles)
	print(df_dict['cifar100'])
	x = ['BALD', 'CoreSet','Entropy', 'Random']
	h_det = sns.heatmap(ax=axn[0], data=df_dict['cifar100'], robust=True, annot=True, cmap=cmap, xticklabels=x,
					yticklabels=x, cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
	axn[0].tick_params(axis='both', which='major', pad=9)
	h_det.set_yticklabels(h_det.get_yticklabels(), rotation = 0, fontsize=50)
	h_det.set_xticklabels(h_det.get_xticklabels(), rotation = 0, fontsize=50)

	m_h = sns.heatmap(ax=axn[1], data=means['cifar100'].transpose(), robust=True, annot=True, cmap=cmap, xticklabels=False, 
					yticklabels=['$\Phi$'], cbar=None, vmin=0, vmax=1, annot_kws={"fontsize": 50}, fmt='.2f')
	axn[1].tick_params(axis='both', which='major', pad=20)
	m_h.set_yticklabels(m_h.get_yticklabels(), rotation = 180, fontsize=50)
	
	fname = f'cifar100_h_t.pdf'
	fpath = plots_path / fname
	print("Saved at: ", fpath)
	os.makedirs(plots_path, exist_ok=True)
	plt.savefig(fpath)

def plot_setting(task_learner, dataset, init_budget, budget, cycles, results_path, plots_path):
	df = pd.DataFrame()
	for fpath in results_path.glob('**/*.txt'):
		# Append extracted results to dataframe
		if 'cifar100_I1000_B2000' in str(fpath):
			_df = read_result_file(fpath)
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
