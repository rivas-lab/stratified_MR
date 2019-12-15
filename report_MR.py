import numpy as np, pandas as pd
from itertools import combinations
from MR import load_MR_data
from scipy.stats import norm
from stratifications import get_stratifications
from utils import z_score_to_p_value

MR_results_dir = 'MR_results'
exposure_name = 'Body mass index (BMI)'
family_history_name = 'Family history of diabetes'
MR_methods = 'MR Egger', 'Inverse variance weighted',
nonsig_only = False
sig_only = False
assert not (nonsig_only and sig_only)

run_names = ('overall', 'family_history', 'PRS',
             'insulin', 'metformin', 'overall_fine_grained')

for name in run_names:
    print('=' * 40)
    print(name)
    print('=' * 40)
    # Set the appropriate settings for this run
    outcome_name = 'type_2_diabetes' if name != 'T1D' else 'type_1_diabetes'
    external_PRS_name = 'diabetes_adj_BMI' if name != 'PRS' else 'diabetes'
    stratify_by_family_history = name == 'family_history'
    unstratified = name in ('overall', 'insulin', 'metformin',
                            'T1D', 'overall_fine_grained')
    subset = name if name in ('metformin', 'insulin') else None
    overall_fine_grained = name == 'overall_fine_grained'
    # Load data
    individuals, exposure, outcome, PRS, family_history, covariate_file = \
        load_MR_data(exposure_name, outcome_name, family_history_name,
                     subset=subset)
    if overall_fine_grained:
        exposure.fine_grained_BMI = True
    # Stratify individuals
    if unstratified:
        stratifications = get_stratifications(exposure=exposure)
    elif stratify_by_family_history:
        stratifications = get_stratifications(
            exposure=exposure, family_history=family_history)
    else:
        stratifications = get_stratifications(
            exposure=exposure, PRS=PRS)
    # Get N_case, N_control and prevalence for each stratification
    N_case = {
        stratification_name: outcome[stratification].sum()
        for stratification_name, stratification in stratifications.items()}
    N_control = {
        stratification_name: stratification.sum() - N_case[stratification_name]
        for stratification_name, stratification in stratifications.items()}
    prevalence = {
        stratification_name: 100 * outcome[stratification].mean()
        for stratification_name, stratification in stratifications.items()}
    # Print prevalences
    for stratification_name, stratification_prevalence in prevalence.items():
        print(f'{stratification_name}: {stratification_prevalence:.1f}%')
    print()
    # Print 2x2 Fisher p-value between each pair of stratifications,
    # based on N_case and N_control
    '''
    if name == 'overall':
        for strat_A, strat_B in combinations(stratifications, 2):
            table = np.array(((N_case[strat_A], N_case[strat_B]),
                              (N_control[strat_A], N_control[strat_B])))
            p = fisher_exact(table)[1]
            if nonsig_only and p < 0.05: continue
            if sig_only and p >= 0.05: continue
            print(f'{strat_A} ({prevalence[strat_A]:.2f}%) vs '
                  f'{strat_B} ({prevalence[strat_B]:.2f}%): Fisher p < {p:.0g}')
        print()
    '''
    # Get MR effect sizes and standard errors for each stratification
    MR_results = {
        stratification_name: pd.read_table(
            f'{MR_results_dir}/{name}/{stratification_name}.MR',
            index_col='method')
        for stratification_name in stratifications}
    MR_betas, MR_sigmas, MR_ps = (
        {stratification_name:
             {MR_method:
                  MR_results[stratification_name].loc[MR_method, key]
              for MR_method in MR_methods}
         for stratification_name in stratifications}
        for key in ('b', 'se', 'pval'))
    # Print MR effect sizes, 95% CIs and standard errors
    # as well as absolute risks
    for MR_method in MR_methods:
        print(f'{MR_method}:')
        for stratification_name in stratifications:
            prev = prevalence[stratification_name]
            beta = MR_betas[stratification_name][MR_method]
            sigma = MR_sigmas[stratification_name][MR_method]
            lower_ci, upper_ci = norm(beta, sigma).ppf((0.025, 0.975))
            exp_beta, exp_lower_ci, exp_upper_ci = map(
                np.exp, (beta, lower_ci, upper_ci))
            p = MR_ps[stratification_name][MR_method]
            beta_abs_risk_red = prev - prev / exp_beta
            lower_ci_abs_risk_red = prev - prev / exp_lower_ci
            upper_ci_abs_risk_red = prev - prev / exp_upper_ci
            print(f'{stratification_name}: {exp_beta:.2f} '
                  f'[{exp_lower_ci:.2f}, {exp_upper_ci:.2f}]; abs '
                  f'{beta_abs_risk_red:.3f} [{lower_ci_abs_risk_red:.3f}, '
                  f'{upper_ci_abs_risk_red:.3f}] '
                  f'({beta:.2g} ± {sigma:.1g}, p = {p:.0g})')
        print()
    print()
    # Print z-score-difference p-value between each pair of stratifications,
    # based on MR betas and sigmas
    for MR_method in MR_methods:
        print(f'{MR_method}:')
        for strat_A, strat_B in combinations(stratifications, 2):
            beta_A = MR_betas[strat_A][MR_method]
            beta_B = MR_betas[strat_B][MR_method]
            sigma_A = MR_sigmas[strat_A][MR_method]
            sigma_B = MR_sigmas[strat_B][MR_method]
            # https://stats.stackexchange.com/a/99536/54578
            Z = (beta_A - beta_B) / np.sqrt(sigma_A ** 2 + sigma_B ** 2)
            p = z_score_to_p_value(Z)
            if nonsig_only and p < 0.05: continue
            if sig_only and p >= 0.05: continue
            print(f'{strat_A} ({np.exp(beta_A):.3f}) vs '
                  f'{strat_B} ({np.exp(beta_B):.3f}): '
                  f'difference p = {p:.2g}, beta_diff = {beta_A - beta_B:.2f}')
        print()
    print()
    # For overweight and obese, what is the HR for 5/10/20 lb weight loss?
    # Assume a roughly average-height individual 1.7m tall (~5'7")
    if name == 'overall':
        ci = lambda beta, sigma: norm(beta, sigma).ppf((0.025, 0.975))
        height = 1.7
        for MR_method in MR_methods:
            print(f'{MR_method}:')
            for amt_weight_loss in 5, 10, 20, 50:
                for lb in False, True:
                    unit = 'lb' if lb else 'kg'
                    def rel_percent(amt_weight_loss, MR_beta, lb=True):
                        if lb:
                            lb_weight_loss = amt_weight_loss
                            kg_weight_loss = lb_weight_loss * 0.453592
                        else:
                            kg_weight_loss = amt_weight_loss
                        BMI_change = kg_weight_loss / height ** 2
                        HR = np.exp(MR_beta) ** -BMI_change
                        percent_reduced_risk = 100 * (1 - HR)
                        return percent_reduced_risk
                    betas = {}
                    abs_risk_red = {}
                    for key in 'overweight', 'obese':
                        stratification_name = f'{key}_exposure'
                        beta = MR_betas[stratification_name][MR_method]
                        sigma = MR_sigmas[stratification_name][MR_method]
                        prev = prevalence[stratification_name]
                        lower_ci, upper_ci = ci(beta, sigma)
                        betas[key, 'mean'] = beta
                        betas[key, 'lower'] = lower_ci
                        betas[key, 'upper'] = upper_ci
                        # Also get abs risk red
                        if lb:
                            lb_weight_loss = amt_weight_loss
                            kg_weight_loss = lb_weight_loss * 0.453592
                        else:
                            kg_weight_loss = amt_weight_loss
                        BMI_change = kg_weight_loss / height ** 2
                        for beta_type in 'mean', 'lower', 'upper':
                            abs_risk_red[key, beta_type] = \
                                prev - prev / np.exp(betas[key, beta_type]) ** \
                                              BMI_change
                    relative_risk_red = \
                        {key: rel_percent(amt_weight_loss, beta, lb)
                         for key, beta in betas.items()}
                    print(f'{amt_weight_loss} {unit} weight loss: '
                          f'overweight relative '
                          f'{relative_risk_red["overweight", "mean"]:.0f}% '
                          f'[{relative_risk_red["overweight", "lower"]:.0f}, '
                          f'{relative_risk_red["overweight", "upper"]:.0f}], '
                          f'abs {abs_risk_red["overweight", "mean"]:.3f}% '
                          f'[{abs_risk_red["overweight", "lower"]:.3f}, '
                          f'{abs_risk_red["overweight", "upper"]:.3f}], '
                          f'obese relative '
                          f'{relative_risk_red["obese", "mean"]:.0f}% '
                          f'[{relative_risk_red["obese", "lower"]:.0f}, '
                          f'{relative_risk_red["obese", "upper"]:.0f}], '
                          f'abs {abs_risk_red["obese", "mean"]:.3f}% '
                          f'[{abs_risk_red["obese", "lower"]:.3f}, '
                          f'{abs_risk_red["obese", "upper"]:.3f}]')
        print()
    # Looking at it the other way around:
    # How much weight do you need to lose to reduce your risk 25%/50%/75%
    if name == 'overall':
        height = 1.7
        for MR_method in MR_methods:
            print(f'{MR_method}:')
            for percent_reduced_risk in 25, 50, 75, 90:
                for lb in False, True:
                    unit = 'lb' if lb else 'kg'
                    def amt_weight_loss(percent_reduced_risk, MR_beta, lb=True):
                        HR = 1 - percent_reduced_risk / 100
                        BMI_change = -np.log(HR) / MR_beta
                        kg_weight_loss = BMI_change * height ** 2
                        if lb:
                            lb_weight_loss = kg_weight_loss / 0.453592
                            return lb_weight_loss
                        else:
                            return kg_weight_loss
                    betas = {}
                    for key in 'overweight', 'obese':
                        stratification_name = f'{key}_exposure'
                        beta = MR_betas[stratification_name][MR_method]
                        sigma = MR_sigmas[stratification_name][MR_method]
                        lower_ci, upper_ci = ci(beta, sigma)
                        betas[key, 'mean'] = beta
                        betas[key, 'lower'] = lower_ci
                        betas[key, 'upper'] = upper_ci
                    amts = {key: amt_weight_loss(percent_reduced_risk, beta, lb)
                            for key, beta in betas.items()}
                    print(f'{percent_reduced_risk:.0f}% reduced risk: '
                          f'overweight '
                          f'{amts["overweight", "mean"]:.0f} {unit} '
                          f'weight loss '
                          f'[{amts["overweight", "lower"]:.0f}, '
                          f'{amts["overweight", "upper"]:.0f}], '
                          f'obese '
                          f'{amts["obese", "mean"]:.0f} {unit} '
                          f'weight loss '
                          f'[{amts["obese", "lower"]:.0f}, '
                          f'{amts["obese", "upper"]:.0f}]')
    # And the same thing using absolute risk
    if name == 'overall':
        height = 1.7
        for MR_method in MR_methods:
            print(f'{MR_method}:')
            for percent_abs_reduced_risk in 1, 2, 5:
                for lb in False, True:
                    unit = 'lb' if lb else 'kg'
                    def amt_weight_loss(percent_reduced_risk, MR_beta, lb=True):
                        HR = 1 - percent_reduced_risk / 100
                        BMI_change = -np.log(HR) / MR_beta
                        kg_weight_loss = BMI_change * height ** 2
                        if lb:
                            lb_weight_loss = kg_weight_loss / 0.453592
                            return lb_weight_loss
                        else:
                            return kg_weight_loss
                    betas = {}
                    percent_reduced_risk = {}
                    for key in 'overweight', 'obese':
                        stratification_name = f'{key}_exposure'
                        beta = MR_betas[stratification_name][MR_method]
                        sigma = MR_sigmas[stratification_name][MR_method]
                        lower_ci, upper_ci = ci(beta, sigma)
                        betas[key, 'mean'] = beta
                        betas[key, 'lower'] = lower_ci
                        betas[key, 'upper'] = upper_ci
                        prev = prevalence[stratification_name]
                        percent_reduced_risk[key] = \
                            100 * percent_abs_reduced_risk / prev
                    amts = {key: amt_weight_loss(
                        percent_reduced_risk[key[0]], beta, lb)
                        for key, beta in betas.items()}
                    print(f'{percent_abs_reduced_risk:.0f}% abs reduced risk: '
                          f'overweight '
                          f'{amts["overweight", "mean"]:.0f} {unit} '
                          f'weight loss '
                          f'[{amts["overweight", "lower"]:.0f}, '
                          f'{amts["overweight", "upper"]:.0f}], '
                          f'obese '
                          f'{amts["obese", "mean"]:.0f} {unit} '
                          f'weight loss '
                          f'[{amts["obese", "lower"]:.0f}, '
                          f'{amts["obese", "upper"]:.0f}]')
