import  matplotlib.pyplot as plt, numpy as np, pandas as pd
from MR import load_instruments, load_instrument_genotypes, load_MR_data
from rpy2.robjects import globalenv, pandas2ri, r
from scipy.stats import pearsonr
pandas2ri.activate()

# Non-linear MR: https://github.com/jrs95/nlmr

try:
    r.library('nlmr')
except:
    print('Warning: nlme not installed; installing now')
    r.library('devtools')
    r.install_github('jrs95/nlmr')
    r.library('nlmr')

r.library('metafor')

# Settings

MR_results_dir = 'MR_results'
exposure_name = 'Body mass index (BMI)'
outcome_name = 'type_2_diabetes'
family_history_name = 'Family history of diabetes'
external_PRS_name = 'Anubha_diabetes'

# Load exposure and outcome

individuals, exposure, outcome, PRS, family_history, covariate_file = \
    load_MR_data(exposure_name, outcome_name, family_history_name,
                 external_PRS_name)

# Load covariates

covariates = pd.read_table(covariate_file, index_col='FID')
covariates.dropna(inplace=True)

# Load Locke et al. sumstats

print('Loading Locke et al. sumstats...')
locke = pd.read_table('/oak/stanford/groups/mrivas/projects/gwassummarystats/'
                      'summary_stats/gwas/giant/BMI.SNPadjSMK.CombinedSexes.'
                      'EuropeanOnly.txt', index_col='rs_id',
                      usecols=['rs_id', 'effect', 'allele_1', 'allele_2'])
locke = locke[~locke.index.duplicated()]

# Create instrument: allele score of 57 BMI SNPs,
# weighted by Locke et al. effect size

print('Loading instruments...')
instruments, instrument_chroms = load_instruments(exposure_name)
instruments = instruments[instruments != 'rs9581854']
assert len(instruments) == 57 - 1
instrument_genotypes = load_instrument_genotypes(
    instruments, instrument_chroms, exposure)

print('Creating allele score...')

# Get Locke and Biobank alleles

locke_alleles_and_effect = locke.reindex(instrument_genotypes.columns)
locke_alleles = locke_alleles_and_effect[['allele_1', 'allele_2']]
all_biobank_alleles = pd.read_csv(
    'imputed_bims/chrALL.bim', sep='\t', usecols=[1, 4, 5], index_col=0,
    header=None)
biobank_alleles = all_biobank_alleles[all_biobank_alleles.index.isin(
    instrument_genotypes.columns)]

# Verify we don't need to strand-flip

quadruplets = locke_alleles.join(biobank_alleles)
strand_flip = quadruplets.iloc[:, 0] != quadruplets.iloc[:, 3]  # [:, 3] is REF
strand_flip = np.where(strand_flip, -1, 1)
locke_effect = locke_alleles_and_effect.effect.reindex(quadruplets.index)
locke_sign = np.sign(locke_effect) * strand_flip
biobank_sign = pd.Series(
    np.sign([pearsonr(instrument_genotypes[rs], exposure)[0]
             for rs in quadruplets.index]), index=quadruplets.index)
assert (locke_sign == biobank_sign).all()

# Get allele score

locke_weights = np.abs(locke_effect) * locke_sign
allele_score = instrument_genotypes.dot(locke_weights.values[:, None]).squeeze()
print(f'pearson(exposure, allele_score) = '
      f'{pearsonr(exposure, allele_score)[0]}')

# Standardize allele score

print('Standardizing allele score...')
allele_score -= allele_score.mean()
allele_score /= allele_score.std()

# Join

print('Subsetting data to individuals in common...')
data = exposure.to_frame().join(outcome, how='inner').join(
    allele_score, how='inner').join(covariates, how='inner')
exposure = data[exposure.name]
outcome = data[outcome.name]
allele_score = data[allele_score.name]
covariates = data.drop([exposure.name, outcome.name, allele_score.name], axis=1)

# Non-linear MR

def non_linear_MR(exposure, outcome, covariates, allele_score, num_quantiles,
                  suffix=''):
    # Get BMI quantiles
    globalenv['covar'] = covariates
    processed_covariates = r(
        'as.matrix(model.matrix(as.formula('
        'paste("~ ", paste(names(covar),collapse=" + "))), data=covar)[,-1])')
    iv_free = r.iv_free(outcome, exposure, allele_score, processed_covariates,
                        num_quantiles, 'binomial')
    IV_free_quantiles = pandas2ri.ri2py(iv_free.rx2('x0q'))
    mean_BMIs = exposure.groupby(IV_free_quantiles).mean()
    # Run non-linear MR
    method = 'Fractional polynomial'
    print(f'{method}, {num_quantiles} quantiles:')
    if method == 'Fractional polynomial':
        mr_func = r.fracpoly_mr
    else:
        mr_func = r.piecewise_mr
    MR_results = mr_func(outcome, exposure, allele_score, covariates,
                         'binomial', num_quantiles)
    # Report betas and sigmas in each stratification
    lace = np.asarray(MR_results.rx2('lace'))
    betas = lace[:, 0]
    sigmas = lace[:, 1]
    for beta, sigma, mean_BMI in zip(betas, sigmas, mean_BMIs):
        print(f'BMI = {mean_BMI:.2f}: '
              f'OR = {np.exp(beta):.3f} '
              f'[{np.exp(beta - 1.96 * sigma):.3f}, '
              f'{np.exp(beta + 1.96 * sigma):.3f}]')
    # Report p-value for Cochran Q heterogeneity test
    # (this is testing "the assumption that the effect of the IV on BMI
    # is constant over the entire distribution of BMI", i.e. testing
    # whether it's valid to apply the method)
    p_het_Q, p_het_trend = MR_results.rx2('p_heterogeneity')
    print(f'IV-BMI heterogeneity: p_het_Q = {p_het_Q:.0g}, '
          f'p_het_trend = {p_het_trend:.0g}')
    # Report p-value for non-linearity (the actual result)
    p_Q = MR_results.rx2('p_tests')[-1]
    print(f'Main result: Cochran Q test p = {p_Q:.0g}')
    # Plot MR estimate per percentile, with 95% CIs as error bars
    plt.axhline(np.exp(betas).mean(), color='0.75', zorder=-1)  # avg odds ratio
    plt.errorbar(mean_BMIs, np.exp(betas),
                 fmt='o', markersize=2, c='k', ecolor='k',
                 yerr=[np.exp(betas) - np.exp(betas - 1.96 * sigmas),
                       np.exp(betas + 1.96 * sigmas) - np.exp(betas)])
    plt.xlabel('BMI')
    plt.ylabel('Odds ratio')
    # Save figure
    plt.savefig(f'non_linear_MR_{num_quantiles}_quantiles{suffix}', dpi=300)
    plt.close()

print('Overall...')
non_linear_MR(exposure, outcome, covariates, allele_score, num_quantiles=50)

# Repeat stratified by PRS tertiles

from stratifications import stratify_by_tertiles
PRS_tertiles = stratify_by_tertiles(PRS)
for tertile_name, tertile_individuals in PRS_tertiles.items():
    print(f'Tertile "{tertile_name}" of PRS...')
    non_linear_MR(exposure.loc[tertile_individuals],
                  outcome.loc[tertile_individuals],
                  covariates.loc[tertile_individuals],
                  allele_score.loc[tertile_individuals],
                  num_quantiles=50, suffix=f'_{tertile_name}')
