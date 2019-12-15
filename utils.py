import numpy as np, os, pandas as pd, subprocess
from collections import Counter
from itertools import tee
# noinspection PyUnresolvedReferences
from pgenlib import PgenReader
from scipy.stats import norm
from sklearn.preprocessing import Imputer
from time import sleep


def escape_string(string):
    substitutions = ((' / ', '_'),
                     ('/', '_'),
                     (' ', '_'),
                     ('(', ''),
                     (')', ''),
                     (';_', '_'))
    for substitution in substitutions:
        string = string.replace(*substitution)
    string = string.rstrip('.')  # "Alcohol intake frequency."
    return string


def load_SNP(rs_number, chrom, genotype_cache=None):
    if isinstance(chrom, str) and chrom.startswith('chr'):
        chrom = chrom[3:]
    if genotype_cache is not None and rs_number in genotype_cache:
        dosages = genotype_cache[rs_number]
        # Make sure we don't re-load after dealloc
        assert dosages is not None
    else:
        bgen = f'/oak/stanford/projects/ukbb/genotypes/' \
               f'pgen_app1372_hrc/ukb_imp_chr{chrom}_v2.mac1.hrc'
        all_sites_file = f'/oak/stanford/projects/ukbb/genotypes/' \
                         f'pgen_app1372_hrc/ukb_imp_chr{chrom}_v2.all_sites.txt'
        dosages = load_dosages(
            bgen, all_sites_file, rs_numbers=[rs_number])
        dosages = dosages.squeeze()
        if genotype_cache is not None:
            genotype_cache[rs_number] = dosages
    return dosages


def is_non_empty_file(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0


def run_GWAS_plink2(GWAS_name, phenotypes, covariate_file,
                    chroms=tuple(range(1, 23)),
                    GWAS_results_dir='GWAS_results',
                    individuals=None, variants=None,
                    force_linear_regression=False,
                    LD_prune=False, extra_options='', verbose=False):
    # Run unimputed Biobank GWAS with covariates, using phenotypes
    # from phe_file and on only the individuals in individuals.
    # Return summary stats.
    assert ' ' not in GWAS_name, f'GWAS name "{GWAS_name}" contains a space!'
    GWAS_results_dir += f'/{GWAS_name}'
    os.makedirs(GWAS_results_dir, exist_ok=True)
    if individuals is not None:
        assert isinstance(individuals, np.ndarray)
    if variants is not None:
        assert isinstance(variants, pd.Series)
    # Get the abspath of the covariate file before we cd
    covariate_file = os.path.abspath(covariate_file)
    # cd to GWAS_results_dir but remember cwd so we can cd back at the end
    original_cwd = os.getcwd()
    os.chdir(GWAS_results_dir)
    # Run GWAS in GWAS_results_dir
    sumstats_cache = 'sumstats.tsv'
    if phenotypes.is_binary:
        # Print # cases and controls
        included_phenotypes = phenotypes[individuals]
        num_cases = (included_phenotypes == 1).sum()
        num_controls = (included_phenotypes == 0).sum()
        indiv_stats = f'binary, {num_cases} cases, {num_controls} controls'
    else:
        indiv_stats = f'qt, {len(individuals)} individuals' \
            if individuals is not None else 'qt'
    if os.path.exists(sumstats_cache):
        print(f'Using cached GWAS {sumstats_cache}, {indiv_stats}...')
        if phenotypes.is_binary:
            assert num_cases >= 25 and num_controls >= 25, \
                f'Only {num_cases} cases and {num_controls} controls!'
        else:
            if individuals is not None:
                # noinspection PyTypeChecker
                assert len(individuals) >= 50
        sumstats = pd.read_table(sumstats_cache)
    else:
        print(f'Running GWAS for {GWAS_name} '
              f'(pwd = {GWAS_results_dir}), {indiv_stats}:')
        if phenotypes.is_binary:
            assert num_cases >= 25 and num_controls >= 25, \
                f'Only {num_cases} cases and {num_controls} controls!'
        else:
            if individuals is not None:
                # noinspection PyTypeChecker
                assert len(individuals) >= 50
            unique_phenotypes = phenotypes.unique()
            assert len(unique_phenotypes) > 1, \
                f'All phenotypes are {unique_phenotypes[0]}!'
            if len(unique_phenotypes) == 2 and not force_linear_regression:
                unique_phenotypes.sort()
                control_value = unique_phenotypes[0]
                case_value = unique_phenotypes[1]
                print(f'WARNING: all phenotypes are {case_value} or '
                      f'{control_value}; converting to case-control '
                      f'where {case_value} = cases, {control_value} = controls')
                # noinspection PyTypeChecker
                phenotypes = pd.Series(np.where(
                    phenotypes == case_value, 1, 0), index=phenotypes.index)
                phenotypes.is_binary = True
        bgen_format = '/oak/stanford/projects/ukbb/genotypes/' \
                      'pgen_app1372_hrc/ukb_imp_chr{}_v2.mac1.hrc'
        # Make individuals file for --keep
        if individuals is not None:
            individual_file = 'individuals.txt'
            pd.DataFrame({'FID': individuals, 'IID': individuals}).to_csv(
                individual_file, sep='\t', index=False, header=False)
            individual_command = f'--keep {individual_file}'
        else:
            individual_command = ''
        # Make variants file for --extract
        if variants is not None:
            variant_file = 'variants.txt'
            variants.to_csv(variant_file, index=False)
            variant_command = f'--extract {variant_file}'
        else:
            variant_command = ''
        # Make fam file for --fam (use chr1 arbitrarily)
        fam_file = '/oak/stanford/groups/mrivas/ukbb24983/imp/pgen/' \
                   'ukb24983_imp_chr1_v3.fam'
        fam_file_with_pheno = 'phenotypes.fam'
        make_fam_file_with_pheno(phenotypes, fam_file, fam_file_with_pheno)
        # Run GWAS on each chromosome
        sumstats_files = {chrom: f'chr{chrom}.PHENO1.glm.logistic'
                                 if phenotypes.is_binary else
                                 f'chr{chrom}.PHENO1.glm.linear'
                          for chrom in chroms}
        frequency_files = {chrom: f'chr{chrom}.afreq' for chrom in chroms}
        for chrom in chroms:
            if is_non_empty_file(sumstats_files[chrom]) and \
                    is_non_empty_file(frequency_files[chrom]):
                print(f'Already ran GWAS on chr{chrom}; continuing...')
                continue
            print(f'Running GWAS on chr{chrom}...')
            genotype_file = bgen_format.format(chrom)
            bim_file = f'/oak/stanford/groups/akundaje/wainberg/Biobank/' \
                       f'imputed_bims/chr{chrom}.bim'
            GWAS_command = (
                f'plink2 {"--silent" if not verbose else ""} '  
                f'--seed 0 '
                f'--pgen {genotype_file}.pgen '
                f'--bim {bim_file} '
                f'--fam {fam_file_with_pheno} '
                f'{individual_command} '
                f'{variant_command} '
                f'--hwe 1e-20 midp --geno '
                f'--glm hide-covar '
                f'--covar {covariate_file} '
                f'--freq '  # to get MAF column
                f'{"--indep-pairwise 500kb 0.2" if LD_prune else ""} '
                f'{extra_options} '
                f'--out chr{chrom}')
            try:
                shell(GWAS_command, raise_if_stderr=True)
            except subprocess.CalledProcessError as e:
                if e.stderr == b'Error: No variants remaining after main ' \
                               b'filters.\n':
                    print('No variants on this chrom; continuing')
                    continue
            if not is_non_empty_file(sumstats_files[chrom]) or \
                    not is_non_empty_file(frequency_files[chrom]):
                raise ValueError(
                    f'GWAS failed! Re-run with:\n'
                    f'cd {GWAS_results_dir}; {GWAS_command}')
            print(f'GWAS run finished for chr{chrom}!')
        print('Aggregating results...')
        sumstats = pd.concat([pd.read_table(sumstats_files[chrom],
                                            delim_whitespace=True)
                              for chrom in chroms
                              if os.path.exists(sumstats_files[chrom])])
        frequencies = pd.concat([pd.read_table(
            frequency_files[chrom], delim_whitespace=True)
            for chrom in chroms if os.path.exists(sumstats_files[chrom])])
        # noinspection PyUnresolvedReferences
        sumstats = sumstats.merge(frequencies[['ID', 'ALT_FREQS']])
        sumstats['samplesize'] = len(phenotypes)
        # A2 is the opposite of A1 (if A1 is ref, it's alt, otherwise it's ref)
        sumstats['A2'] = np.where(sumstats['A1'] == sumstats['REF'],
                                  sumstats['ALT'], sumstats['REF'])
        # Rename columns to their plink 1.9 equivalents
        sumstats.rename(columns={'ID': 'SNP', 'ALT_FREQS': 'MAF'}, inplace=True)
        # Save
        sumstats.to_csv(sumstats_cache, sep='\t', index=False)
        print(f'Done {GWAS_name} GWAS!')
    assert pd.isnull(sumstats).mean().max() < 0.2
    sumstats.dropna(axis=0, inplace=True)
    sumstats.name = GWAS_name
    # cd back to original cwd before we return
    os.chdir(original_cwd)
    return sumstats


def shell(cmd_or_cmds, raise_if_stderr=False, return_stdout=False):
    assert isinstance(cmd_or_cmds, (str, list, tuple))
    if isinstance(cmd_or_cmds, str):
        # Single cmd_or_cmds
        result = subprocess.run(
            cmd_or_cmds, shell=True, executable='/bin/bash',
            stdout=subprocess.PIPE if return_stdout else None,
            stderr=subprocess.PIPE)
        if raise_if_stderr and result.stderr:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr)
        elif result.returncode:
            print(result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr)
        if return_stdout:
            return result.stdout
    else:
        # Multiple commands
        assert all(isinstance(cmd, str) for cmd in cmd_or_cmds)
        processes = [subprocess.Popen(
            cmd, shell=True, executable='/bin/bash',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in
            cmd_or_cmds]
        stdouts = []
        stderrs = []
        for process in processes:
            while process.poll() is None:
                sleep(0.5)
            stdout, stderr = process.communicate()
            stdouts.append(stdout)
            stderrs.append(stderr)
            retcode = process.poll()
            if retcode:
                print(stderr)
                raise subprocess.CalledProcessError(
                    retcode, process.args, output=stdout, stderr=stderr)
        if raise_if_stderr and any(stderrs):
            raise RuntimeError(Counter(stderrs))
        return stdouts


def z_score_to_p_value(z):
    p = 2 * np.where(z > 0, norm.sf(z), norm.cdf(z))
    return p


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)




def make_fam_file_with_pheno(phenotypes, fam_file, fam_file_with_pheno):
    # Make a fam file with phenotypes by joining the phenotypes
    # with the rest of the information from a fam_file without phenotypes.
    # IDs in fam_file not found in phenotypes will still have
    # missing phenotypes in fam_file_with_pheno
    if isinstance(phenotypes, str):
        subprocess.check_call(
            f'set -o pipefail; awk -v OFS="\t" \'NR==FNR{{phe[$1]=$3;next}} '
            f'($1 in phe){{$6=phe[$1]}};1\' {phenotypes} {fam_file} > '
            f'{fam_file_with_pheno}', shell=True, executable='/bin/bash')
    elif isinstance(phenotypes, pd.Series):
        # Load fam file, without phenotype column (which is all -9s)
        fam_info = pd.read_table(fam_file, header=None, index_col=0,
                                 usecols=[0, 1, 2, 3, 4])
        # Add phenotypes as last column
        phenotypes.name = 'phenotypes'
        fam_info = fam_info.join(phenotypes)
        # Replace missing phenotypes by -9
        # Note: can't remove these because plink requires the fam size to match
        # the bed size, otherwise it gives the error
        # "Error: Invalid .bed file size (expected X bytes)."
        fam_info['phenotypes'].fillna(-9, inplace=True)
        assert not (fam_info['phenotypes'] == -9).all()
        if phenotypes.is_binary:
            # The join resets phenotypes to int (because of NaNs), so fix that
            fam_info['phenotypes'] = fam_info['phenotypes'].astype(int)
            fam_info.loc[fam_info['phenotypes'] != -9, 'phenotypes'] += 1
            unique = pd.unique(fam_info['phenotypes'])
            assert np.array_equal(np.sort(unique), (1, 2)) or \
                   np.array_equal(np.sort(unique), (-9, 1, 2))
        # Save new fam file
        fam_info.to_csv(fam_file_with_pheno, header=None, sep='\t')
    else:
        raise ValueError(f'Invalid type {type(phenotypes)} for phenotypes!')


def load_dosages(bgen, all_sites_file, rs_numbers=None, impute_missing=False):
    pgen = PgenReader(f'{bgen}.pgen'.encode())
    num_individuals = pgen.get_raw_sample_ct()
    if rs_numbers is None:
        num_variants = pgen.get_variant_ct()
        dosages = np.empty((num_variants, num_individuals))
        for variant_index in range(num_variants):
            pgen.read_dosages(variant_index, dosages[variant_index])
    else:
        num_variants = len(rs_numbers)
        dosages = np.empty((num_variants, num_individuals))
        for dosage_index, rs_number in enumerate(rs_numbers):
            bp = int(subprocess.check_output(
                f'fgrep "{rs_number}\t" {all_sites_file}',
                shell=True).decode().split()[3])
            variant_index = int(subprocess.check_output(
                f'fgrep -n -m 1 "\t{bp}\t" {bgen}.pvar',
                shell=True).decode().split(':')[0]) - 2
            pgen.read_dosages(variant_index, dosages[dosage_index])
    # Mean-impute missing dosages
    if impute_missing:
        Imputer(missing_values=-9, copy=False).fit_transform(dosages)
    return dosages


def get_covariate_file(townsend=False, optional=False, no_sex=False):
    covariate_file = 'covariates'
    if townsend:
        covariate_file += '_townsend'
    if optional:
        covariate_file += '_with_optional'
    if no_sex:
        covariate_file += '_no_sex'
    covariate_file += '.tsv'
    if not os.path.exists(covariate_file):
        vanilla_covariate_file = '/oak/stanford/groups/mrivas/ukbb/' \
                                 'master_phe/ukb24983_GWAS_covar.phe'
        covariates = pd.read_table(vanilla_covariate_file, index_col='FID')
        assert not covariates.isnull().any().any()
        from phenotype import get_phenotype
        # Include one-hot-encoded assessment center
        # (biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=54)
        assessment_center = get_phenotype('UK Biobank assessment centre')
        one_hot_encoded = pd.get_dummies(assessment_center)
        for center in one_hot_encoded:
            if f'AC_{center}' == 'AC_11009': continue
            covariates[f'AC_{center}'] = one_hot_encoded[center]
        if townsend:
            townsend = get_phenotype(
                'Townsend deprivation index at recruitment',
                negative_values_to_keep='all')
            covariates['SES'] = townsend
        if optional:
            qualifications = get_phenotype(
                'Qualifications', negative_values_to_keep=[-7])
            # Map to 3: college/uni, 2: professional, 1: high school, 0: none
            # http://biobank.ctsu.ox.ac.uk/showcase/field.cgi?id=6138
            covariates['Qualifications'] = qualifications.map({
                1: 3,  # College or University degree
                2: 1,  # A levels/AS levels or equivalent
                3: 1,  # O levels/GCSEs or equivalent
                4: 1,  # CSEs or equivalent
                5: 2,  # NVQ or HND or HNC or equivalent
                6: 2,  # Other professional qualifications eg: nursing, teaching
                -7: 0  # None of the above
            })
        if no_sex:
            del covariates['sex']
        covariates.dropna(axis=0, inplace=True)  # restrict to valid indivs.
        assert len(covariates) > 330000
        covariates.to_csv(covariate_file, sep='\t')
    return covariate_file


def get_Anubha_diabetes_PRS():
    PRS_cache_prefix = 'external_PRS/Anubha_diabetes_PRS'
    PRS_cache = f'{PRS_cache_prefix}.tsv'
    if os.path.exists(PRS_cache):
        print(f'Loading cached Anubha PRS from {PRS_cache}...')
        PRS = pd.read_table(PRS_cache, index_col=0, header=None, squeeze=True)
    else:
        PRS_weights_file = 'external_PRS/T2D.PRS.Mahajan2018.noUKBB.txt.gz'
        white_british_file = \
            '/oak/stanford/groups/mrivas/ukbb24983/sqc/' \
            'population_stratification/ukb24983_white_british.phe'
        chrom_PRSs = {}
        chroms = reversed(range(1, 22))  # NOT 23 - chr22 is missing from PRS
        for chrom in chroms:
            chrom_PRS_prefix = f'{PRS_cache_prefix}_chr{chrom}'
            chrom_PRS_file = f'{chrom_PRS_prefix}.sscore'
            # Compute PRS
            if not os.path.exists(chrom_PRS_file):
                pgen = f'/oak/stanford/projects/ukbb/genotypes/' \
                       f'pgen_app1372_hrc/ukb_imp_chr{chrom}_v2.mac1.hrc.pgen'
                bim = f'/oak/stanford/groups/akundaje/wainberg/Biobank/' \
                      f'imputed_bims/chr{chrom}.bim'
                fam = f'/oak/stanford/groups/mrivas/ukbb24983/imp/pgen/' \
                      f'ukb_imp_chr1_v2.mac1.hrc.fam'
                shell(f'plink2 '
                      f'--pgen {pgen} '
                      f'--bim {bim} '
                      f'--fam {fam} '
                      f'--keep {white_british_file} '
                      f'--score {PRS_weights_file} 1 4 6 header center '
                      f'cols=scoresums '
                      f'--out {chrom_PRS_prefix}')
            # Read PRS
            chrom_PRS = pd.read_table(
                chrom_PRS_file,
                usecols=['#IID', 'SCORE1_SUM'], index_col='#IID',
                delim_whitespace=True, squeeze=True)
            chrom_PRSs[chrom] = chrom_PRS
        # Sum PRS across chromosomes
        chrom_PRSs = pd.DataFrame(chrom_PRSs, columns=chroms)
        assert not chrom_PRSs.isnull().any().any()
        PRS = chrom_PRSs.sum(axis=1)
        PRS.to_csv(PRS_cache, sep='\t')
    PRS.name = 'PRS'
    return -PRS  # sign flip
