import argparse, numpy as np, os, pandas as pd, warnings
from pathlib import Path
from phenotype import get_phenotype, get_phenotype_from_tab_file
# noinspection PyUnresolvedReferences
from rpy2.rinterface import NULL
from rpy2.robjects import pandas2ri, r
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from stratifications import get_stratifications
from utils import escape_string, get_Anubha_diabetes_PRS, get_covariate_file, \
    load_SNP, run_GWAS_plink2
warnings.simplefilter('ignore', UserWarning)  # r.library
r.library('TwoSampleMR')
r.library('ggplot2')
r.source('mr_presso.R')


def load_MR_data(exposure_name, outcome_name, family_history_name, subset=None):
    print('Loading phenotypes...')
    exposure = get_phenotype(exposure_name)
    outcome = get_phenotype(outcome_name)
    print('Loading PRS...')
    PRS = get_Anubha_diabetes_PRS()
    print('Loading family history...')
    family_history = get_phenotype(family_history_name)
    assert family_history.is_binary
    print('Loading latest Biobank individuals...')
    all_individuals = pd.read_table(
        '/oak/stanford/groups/mrivas/ukbb24983/imp/pgen/'
        'ukb24983_imp_chr1_v3.fam', usecols=[0],
        header=None, squeeze=True)
    print(f'Filtering to individuals common to exposure '
          f'(N = {len(exposure)}), outcome (N = {len(outcome)}), '
          f'PRS (N = {len(PRS)}), family history '
          f'(N = {len(family_history)}) and latest Biobank '
          f'individuals (N = {len(all_individuals)})...')
    data = exposure.to_frame().join(outcome, how='inner').join(
        PRS, how='inner').join(family_history, how='inner')
    data = data[data.index.isin(all_individuals.values)]
    if subset is not None:
        print(f'Subsetting to {subset} individuals...')
        covariates = pd.read_table(get_covariate_file(), index_col='FID')
        if subset in ('young', 'old'):
            age = covariates['age']
            subset_mask = age < age.median() if subset == 'young' else \
                age >= age.median()
        elif subset in ('male', 'female'):
            sex = covariates['sex']
            subset_mask = sex == (1 if subset == 'male' else 0)
        elif subset in ('insulin', 'metformin'):
            # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20003
            medications = get_phenotype_from_tab_file(
                'Treatment/medication code', all_phenotype_columns=True)
            # https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=4&nl=1
            medication_code = 1140883066 if subset == 'insulin' else 1140884600
            # For cases, require that medication code
            subset_mask = (medications == medication_code).any(axis=1)
            # but include all controls as usual
            subset_mask |= outcome == 0
        else:
            raise ValueError(f'Unknown subset {subset}!')
        subset_individuals = subset_mask[subset_mask].index
        print(f'{len(subset_individuals)} individuals are {subset}.')
        data = data[data.index.isin(subset_individuals)]
    data[exposure_name].is_binary = exposure.is_binary
    data[outcome_name].is_binary = outcome.is_binary
    exposure, outcome, PRS, family_history = \
        data[exposure_name], data[outcome_name], data['PRS'], \
        data[family_history_name]
    individuals = data.index.values
    print(f'{len(individuals)} individuals are common to all four')
    print('PRS metrics:')
    pearson, pearson_p = pearsonr(outcome, PRS)
    print(f'Pearson({outcome_name}, PRS) = {pearson:.2f} '
          f'(p = {pearson_p:.0g})')
    if outcome.is_binary:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(outcome, PRS)
        print(f'AUC(PRS → {outcome_name}) = {auc:.2f}')
    print('Getting covariate file...')
    # Can't use sex as covariate if stratifying by sex
    covariate_file = get_covariate_file(no_sex=subset in ('male', 'female'))
    return individuals, exposure, outcome, PRS, family_history, covariate_file


def run_MR(exposure_sumstats, outcome_sumstats,
           stratification_name, MR_results_dir, args):
    # Run MR on exposure → outcome using TwoSampleMR
    # Docs: https://mrcieu.github.io/TwoSampleMR/
    # Installation: library(devtools); options(unzip="unzip");
    #               install_github("MRCIEU/TwoSampleMR")
    # Make sure lengths match
    assert len(exposure_sumstats) == len(outcome_sumstats)
    # Make sure we have the right columns
    MR_results_file = f'{MR_results_dir}/{stratification_name}.MR'
    if os.path.exists(MR_results_file):
        print(f'Already ran MR for {stratification_name}!')
        return
    # Perform MR
    print(f'Running MR for {stratification_name}...')
    for sumstats in exposure_sumstats, outcome_sumstats:
        assert isinstance(sumstats, pd.DataFrame)
        assert 'BETA' in sumstats or 'OR' in sumstats
        if 'BETA' not in sumstats:
            sumstats['BETA'] = np.log(sumstats['OR'])
    options = {
        'snp_col': 'SNP', 'beta_col': 'BETA', 'se_col': 'SE', 'eaf_col': 'MAF',
        'effect_allele_col': 'A1', 'other_allele_col': 'A2',
        'pval_col': 'P', 'samplesize_col': 'samplesize'
    }
    for column in options.values():
        assert column in exposure_sumstats
        assert column in outcome_sumstats
    # Get instruments
    if args.verbose: print('Get instruments')
    exposure_dat = r.format_data(
        pandas2ri.py2ri(exposure_sumstats), type='exposure', **options)
    # Get outcome
    if args.verbose: print('Get outcome')
    outcome_dat = r.format_data(
        pandas2ri.py2ri(outcome_sumstats), type='outcome', **options)
    # Harmonise the exposure and outcome data
    if args.verbose: print('Harmonise the exposure and outcome data')
    dat = r.harmonise_data(exposure_dat, outcome_dat, action=1)
    # Perform MR
    if args.verbose: print('Perform MR')
    res_R = r.mr(dat, method_list=r.c(
        'mr_weighted_median', 'mr_egger_regression', 'mr_ivw'))
    # Convert R dataframe to Python
    if args.verbose: print('Convert R dataframe to Python')
    res = pandas2ri.ri2py(res_R)
    res.drop(['id.exposure', 'id.outcome', 'outcome', 'exposure'],
             axis=1, inplace=True)  # delete unnecessary columns
    # Print MR-Egger intercept and p-value
    test = r.mr_pleiotropy_test(dat)
    test = pandas2ri.ri2py(test).squeeze()
    print(f'Egger intercept = {test.egger_intercept:.2f}, p = {test.pval:.0g}')
    # Also run MR-Presso
    raw_MRPresso_results = r.mr_presso(
        BetaOutcome='beta.outcome', BetaExposure='beta.exposure',
        SdOutcome='se.outcome', SdExposure='se.exposure',
        data=dat, OUTLIERtest=True, DISTORTIONtest=True,
        NbDistribution=1000, seed=0)
    MRPresso_results, metadata = map(pandas2ri.ri2py, raw_MRPresso_results)
    del MRPresso_results['Exposure']
    del MRPresso_results['T-stat']
    MRPresso_global_p = metadata.rx2('Pvalue')[0] \
        if metadata.rx2('Pvalue') != NULL else np.nan
    print(f'MR-Presso global p = {MRPresso_global_p:.0g}')
    if MRPresso_global_p < 0.05:
        excluded_SNPs = exposure_sumstats.SNP[
            metadata.rx2('Distortion test').rx2('Outliers Indices')]
        selected_SNPs = exposure_sumstats.SNP[
            ~exposure_sumstats.SNP.isin(excluded_SNPs)]
        MRPresso_results['nsnp'] = len(selected_SNPs)
    else:
        MRPresso_results['nsnp'] = r.nrow(dat)[0]
    MRPresso_results.rename(columns={
        'MR Analysis': 'method', 'Causal Estimate': 'b', 'Sd': 'se',
        'P-value': 'pval'}, inplace=True)
    MRPresso_results['method'] = 'MRPresso ' + MRPresso_results['method']
    # Outlier-corrected may be nan if there are no outliers - fix this now
    MRPresso_results.fillna(MRPresso_results.iloc[0], inplace=True)
    res = pd.concat([res, MRPresso_results])
    assert isinstance(res, pd.DataFrame)
    # Print basic info: beta and CI for each method
    print(f'{stratification_name} ({len(exposure_sumstats)} instruments):')
    for method, beta, sigma in zip(res['method'], res['b'], res['se']):
        lower_ci, upper_ci = norm(beta, sigma).ppf((0.025, 0.975))
        print(f'{method} = {np.exp(beta):.2f} '
              f'[{np.exp(lower_ci):.2f}, {np.exp(upper_ci):.2f}]')
    # Print MRPresso selected SNPs
    if MRPresso_global_p < 0.05:
        print(f'MRPresso selected SNPs: {", ".join(selected_SNPs)}')
    # Save MR results
    print(f'Saving MR results to {MR_results_file}...')
    res.to_csv(MR_results_file, sep='\t', index=False)
    # We're done now!
    print(f'Done MR for {stratification_name}!')


def load_instruments(exposure_name):
    assert exposure_name == 'Body mass index (BMI)'
    # 69 BMI instruments from https://www.bmj.com/content/352/bmj.i582
    # from Supp Table C (https://www.bmj.com/content/bmj/suppl/2016/03/08/
    # bmj.i582.DC1/tyrj029564.ww1_default.pdf)
    # filtered to 57 passing QC
    # rs1558902 is FTO
    instruments = pd.Series({
        'rs7899106': 'chr10', 'rs17094222': 'chr10', 'rs11191560': 'chr10',
        'rs4256980': 'chr11', 'rs2176598': 'chr11', 'rs3817334': 'chr11',
        'rs12286929': 'chr11', 'rs7138803': 'chr12', 'rs11057405': 'chr12',
        'rs9581854': 'chr13', 'rs12429545': 'chr13', 'rs10132280': 'chr14',
        'rs12885454': 'chr14', 'rs7141420': 'chr14', 'rs3736485': 'chr15',
        'rs758747': 'chr16', 'rs12446632': 'chr16', 'rs2650492': 'chr16',
        'rs1558902': 'chr16', 'rs1000940': 'chr17', 'rs12940622': 'chr17',
        'rs1808579': 'chr18', 'rs7243357': 'chr18', 'rs6567160': 'chr18',
        'rs17724992': 'chr19', 'rs29941': 'chr19', 'rs2287019': 'chr19',
        'rs657452': 'chr1', 'rs3101336': 'chr1', 'rs17024393': 'chr1',
        'rs543874': 'chr1', 'rs2820292': 'chr1', 'rs13021737': 'chr2',
        'rs11126666': 'chr2', 'rs1016287': 'chr2', 'rs11688816': 'chr2',
        'rs2121279': 'chr2', 'rs1528435': 'chr2', 'rs7599312': 'chr2',
        'rs6804842': 'chr3', 'rs3849570': 'chr3', 'rs13078960': 'chr3',
        'rs16851483': 'chr3', 'rs1516725': 'chr3', 'rs10938397': 'chr4',
        'rs11727676': 'chr4', 'rs2112347': 'chr5', 'rs2207139': 'chr6',
        'rs13191362': 'chr6', 'rs1167827': 'chr7', 'rs17405819': 'chr8',
        'rs2033732': 'chr8', 'rs4740619': 'chr9', 'rs10968576': 'chr9',
        'rs6477694': 'chr9', 'rs1928295': 'chr9', 'rs10733682': 'chr9'
    })
    assert len(instruments) == 57
    instrument_chroms = pd.Series(instruments.values)
    instruments = pd.Series(instruments.index)
    return instruments, instrument_chroms


def load_instrument_genotypes(instruments, instrument_chroms, exposure):
    instrument_genotypes = np.array([
        load_SNP(instrument, chrom) for instrument, chrom
        in zip(instruments, instrument_chroms)]).T
    all_individuals = pd.read_table(
        '/oak/stanford/groups/mrivas/ukbb24983/imp/pgen/'
        'ukb24983_imp_chr1_v3.fam', usecols=[0],
        header=None, squeeze=True)
    assert exposure.index.isin(all_individuals.values).all()
    instrument_genotypes = pd.DataFrame(
        instrument_genotypes, index=all_individuals, columns=instruments)
    instrument_genotypes = instrument_genotypes.reindex(exposure.index)
    return instrument_genotypes


def stratified_MR(args):
    GWAS_results_dir = args.name
    MR_results_dir = f'MR_results/{GWAS_results_dir}'
    os.makedirs(MR_results_dir, exist_ok=True)
    MR_done_file = f'{MR_results_dir}/MR.done'
    MR_done = os.path.exists(MR_done_file)
    if MR_done:
        print(f'{args.name} ({args.exposure_name} → {args.outcome_name}) '
              f'already done!')
    else:
        print(f'Running MR on {args.exposure_name} → {args.outcome_name}...')
        print('Loading data...')
        individuals, exposure, outcome, PRS, family_history, covariate_file = \
            load_MR_data(args.exposure_name, args.outcome_name,
                         args.family_history_name, args.subset)
        print('Loading instruments...')
        instruments, instrument_chroms = load_instruments(args.exposure_name)
        print('Loading instrument genotypes...')
        instrument_genotypes = load_instrument_genotypes(
            instruments, instrument_chroms, exposure)
        print('Correcting for collider bias...')
        # Exposure
        predicted_exposure = LinearRegression().fit(
            instrument_genotypes, exposure).predict(instrument_genotypes)
        exposure_r2 = r2_score(exposure, predicted_exposure)
        predicted_exposure -= predicted_exposure.mean()
        instrument_free_exposure = exposure - predicted_exposure
        print(f'exposure r^2 = {exposure_r2:.3f}')
        # PRS
        predicted_PRS = LinearRegression().fit(
            instrument_genotypes, PRS).predict(instrument_genotypes)
        PRS_r2 = r2_score(PRS, predicted_PRS)
        predicted_PRS -= predicted_PRS.mean()
        instrument_free_PRS = PRS - predicted_PRS
        print(f'PRS r^2 = {PRS_r2:.3f}')
        # Family history
        # (use unpenalized LR)
        # noinspection PyUnresolvedReferences
        predicted_FH = LogisticRegression(
            penalty='l2', C=np.inf, solver='lbfgs').fit(
            instrument_genotypes, family_history).predict_proba(
            instrument_genotypes)[:, 1]
        predicted_FH_decile = (np.argsort(
            predicted_FH) // (len(predicted_FH) / 10)).astype(int)
        FH_deciles = pd.Series(predicted_FH_decile[family_history == 1])
        no_FH_deciles = pd.Series(predicted_FH_decile[family_history == 0])
        FH_value_counts = FH_deciles.value_counts()
        no_FH_value_counts = no_FH_deciles.value_counts()
        for decile in FH_value_counts.index:
            # Subsample no family history to 4.5x the size
            ratio = 4.5
            assert ((no_FH_value_counts / FH_value_counts) >= ratio).all()
            assert no_FH_value_counts[decile] >= ratio * FH_value_counts[decile]
            no_FH_decile_indices = np.flatnonzero(
                (family_history == 0) & (predicted_FH_decile == decile))
            subsample_mask = np.zeros_like(no_FH_decile_indices, dtype=bool)
            subsample_mask[:int(no_FH_value_counts[decile] -
                                ratio * FH_value_counts[decile])] = True
            np.random.seed(0)
            np.random.shuffle(subsample_mask)
            family_history.iloc[no_FH_decile_indices[subsample_mask]] = np.nan
        print(f'After subsampling: FH N = {(family_history == 1).sum()}, '
              f'no FH N = {(family_history == 0).sum()}, '
              f'excluded N = {family_history.isnull().sum()}')
        print('Stratifying individuals...')
        if args.fine_grained_BMI:
            instrument_free_exposure.fine_grained_BMI = True
        if args.unstratified:
            stratifications = get_stratifications(
                exposure=instrument_free_exposure)
        elif args.stratify_by_family_history:
            stratifications = get_stratifications(
                exposure=instrument_free_exposure,
                family_history=family_history)
        else:
            stratifications = get_stratifications(
                exposure=instrument_free_exposure, PRS=instrument_free_PRS)
        print('Running GWAS/loading cached GWAS...')
        # For exposure GWAS, only use outcome controls
        # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5082560/
        sumstats = {
            stratification_name: {
                phenotype_name:
                    run_GWAS_plink2(
                        GWAS_name=os.path.join(
                            GWAS_results_dir,
                            escape_string(f'{phenotype_name}_'
                                          f'{stratification_name}')),
                        phenotypes=phenotypes,
                        covariate_file=covariate_file,
                        individuals=individuals[stratification]
                        if phenotype_name == args.outcome_name else
                        individuals[stratification & (outcome == 0)],
                        variants=instruments,
                        verbose=args.verbose)
                for phenotype_name, phenotypes in (
                (args.exposure_name, exposure), (args.outcome_name, outcome))}
            for stratification_name, stratification in stratifications.items()
            # Only run on stratifications (exposure-PRS quadrants) if
            # there are at least 100 cases in that stratification.
            if (outcome[individuals[stratification]] == 1).sum() >= 100}
        print('Running MR...')
        for stratification_name in stratifications:
            run_MR(
                exposure_sumstats=sumstats[
                    stratification_name][args.exposure_name],
                outcome_sumstats=sumstats[
                    stratification_name][args.outcome_name],
                stratification_name=stratification_name,
                MR_results_dir=MR_results_dir,
                args=args)
        print(f'Done MR for {args.exposure_name} → {args.outcome_name}!')
        Path(MR_done_file).touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--exposure-name', required=True)
    parser.add_argument('--outcome-name', required=True)
    parser.add_argument('--family-history-name', required=True)
    parser.add_argument('--unstratified', action='store_true')
    parser.add_argument('--stratify-by-family-history', action='store_true')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--fine-grained-BMI', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    stratified_MR(args)

