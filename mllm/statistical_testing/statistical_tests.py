import pingouin
from pingouin import normality
from scipy.stats import percentileofscore, pearsonr, kstest, mannwhitneyu, kruskal, t, normaltest, wilcoxon
import numpy as np
import pingouin as pg


# My own implementation, not used at the end
def paired_student_test(x, y):
	"""
	:param x: first sample.
	:param y: second sample. We expect x and y to have matching indexes.
	:return: p-value of the two-sided test
	"""
	d = np.array(x) - np.array(y)
	mean_d = np.mean(d)
	std_d = np.std(d, ddof=1) # to force the expression of sample std with n-1 denominator
	n = len(x)
	t_stat = mean_d / (std_d / np.sqrt(n))
	cdf = t.cdf(t_stat, df=(n-1)) # P(T <= t_stat)
	symetric_cdf = 1 - cdf # for two-sided test
	lower_cdf, upper_cdf = min(cdf, symetric_cdf), max(cdf, symetric_cdf)
	p_value = lower_cdf + (1 - upper_cdf)
	return p_value


def normal_kolmogorov_smirnov(x):
	return kstest(rvs=(x - x.mean()) / (x.std()), cdf="norm", alternative="two-sided")[1]


def two_sample_equal_mean_rejected(x, y, x_name, y_name, paired, alpha=0.05, verbose=True):
	"""
	Tests whether the two samples x and y have equal mean :
	if their differences are normally distributed, we use the paired Student test ;
	else we use the Wilcoxon signed-rank test
	:param x: first sample.
	:param y: second sample. We expect x and y to have matching indexes.
	:param x_name:
	:param y_name:
	:param alpha: the alpha-value that is used for all tests
	:return: True if the null hypothesis is rejected (in other words the difference is significant), else False
	"""

	# We test normality assumption
	if paired:
		d = np.array(x) - np.array(y)
		d_normal_p_value = normal_kolmogorov_smirnov(d)
		normality_checked = d_normal_p_value < alpha
	else:
		x_normal_p_value = normal_kolmogorov_smirnov(np.array(x))
		x_normality_checked = x_normal_p_value < alpha
		y_normal_p_value = normal_kolmogorov_smirnov(np.array(y))
		y_normality_checked = y_normal_p_value < alpha
		normality_checked = x_normality_checked and y_normality_checked

	if normality_checked:
		pingouin_res = pingouin.ttest(x, y, paired=paired, alternative='two-sided', correction='auto', confidence=1-alpha)
		p_value = pingouin_res.at["T-test", "p-val"]
		if verbose:
			print(f"Normality assumption verified for comparing {x_name} and {y_name}. -> Student test"
				  #f"\nPaired Student test p-value : {p_value}"
				  )

	else:
		if paired:
			p_value = wilcoxon(x, y)[1]
			if verbose:
				print(f"Normality assumption not verified for comparing {x_name} and {y_name} (paired). -> Wilcoxon test"
					  #f"\nWilcoxon signed-rank test p-value : {p_value}"
					  )
		else:
			p_value = mannwhitneyu(x, y)[1]
			if verbose:
				print(
					f"Normality assumption not verified for comparing {x_name} and {y_name} (not paired). -> Mann-Whiteney U rank test"
					# f"\nMann-Whiteney U rank test p-value : {p_value}"
					)
	return p_value < alpha, round(np.mean(x) - np.mean(y), 3)


def all_groups_normal(df, factor_str, target_variable, alpha=0.05):
	# We check the normal distribution within each group
	# N.B.: pingouin also proposes normality tests, but we remain consistent using Kolmogorov-Smirnov test
	all_normal = True
	for group in df[factor_str].unique():
		group_df = df[df[factor_str] == group]
		ks_stat = normal_kolmogorov_smirnov(group_df[target_variable])
		all_normal = all_normal and (ks_stat >= alpha)
	return all_normal


def anova_equal_mean_rejected(df, factors, target_variable, repeated_measure_subject=None, repeated_measure=True, alpha=0.05):
	"""

	:param df: the dataframe where rows correspond to samples : if repeated_measuring, we expect matching between columns
	:param factors: names of the columns used as factors
	(the fact it is 1-way or multi-way ANOVA is deduced from the number of factors)
	:param repeated_measure: whether we use repeated measure
	:return:
	"""
	# We define the useful grouping variable
	factor_str = factors[0] if len(factors) == 1 else " * ".join(factors)
	if len(factors) > 1:
		df[factor_str] = df[factors].apply(lambda x: ','.join(x.astype(str)), axis=1)

	# We check the normal distribution within each group
	# N.B.: pingouin also proposes normality tests, but we remain consistent using Kolmogorov-Smirnov test
	all_normal = True
	for group in df[factor_str].unique():
		group_df = df.query(f"{factor_str} == @group")
		ks_stat = normal_kolmogorov_smirnov(group_df[target_variable])
		all_normal = all_normal and (ks_stat >= alpha)
	if all_normal:
		print(f"Normality assumption (of {target_variable} over {factor_str}) verified")
	else:
		print(f"Normality assumption (of {target_variable} over {factor_str}) not verified")

	if repeated_measure:
		# Assumptions : normality, sphericity
		try:
			sphericity_test = pg.sphericity(df, dv=target_variable, subject=repeated_measure_subject,
						  within=factors)[-1]
			sphericity_checked = sphericity_test >=alpha
			if sphericity_checked:
				print(f"Sphericity assumption (of {target_variable} over {factor_str}, for subject {repeated_measure_subject}) verified")
			else:
				print(
					f"Sphericity assumption (of {target_variable} over {factor_str}, for subject {repeated_measure_subject}) not verified")
		except:
			print(f"(sphericity test not supported in this case (of {target_variable} over {factor_str}, for subject {repeated_measure_subject}))")

		res = pingouin.rm_anova(data=df, dv=target_variable, within=factors, subject=repeated_measure_subject, detailed=True)
	else:
		# Assumptions : independence, normality, homoscedasticity
		# First we test homoscedasticity (but for now, not prepared if it is not verified)
		homoscedasticity_checked = pg.homoscedasticity(data=df, dv=target_variable, group=factor_str, alpha=alpha).at['levene', 'equal_var']
		if homoscedasticity_checked:
			print(f"Homoscedasticity assumption (of {target_variable} over {factor_str}) verified")
		else:
			print(f"Homoscedasticity assumption (of {target_variable} over {factor_str}) not verified")

		res = pingouin.anova(data=df, dv=target_variable, between=factors, detailed=True)

	res.set_index('Source', inplace=True)
	p_value = res.query("Source == @factor_str").at[factor_str, "p-unc"]
	print(f"p_value of ANOVA test : {p_value}")

	return p_value < alpha
