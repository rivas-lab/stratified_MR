import numpy as np, os, pandas as pd, subprocess


def get_phenotype_table(cache=[]):
    if not cache:
        phenotype_table = pd.read_table(
            'Data_Dictionary_Showcase_with_TableID.tsv',
            usecols=['Field', 'FieldID', 'TableID'], index_col='Field')
        phenotype_table = phenotype_table[phenotype_table['TableID'].notnull()]
        assert not pd.isnull(phenotype_table).any().any()
        cache.append(phenotype_table)
    phenotype_table, = cache
    return phenotype_table


def get_full_field_IDs(table_ID, cache={}):
    # e.g. 'f.20110.1.2'
    if table_ID in cache:
        suffixed_field_IDs = cache[table_ID]
    else:
        if table_ID == 9797:
            tab_file = '/oak/stanford/groups/mrivas/ukbb/24983/' \
                       'phenotypedata/9796/9797/download/ukb9797.tab'
        else:
            raise ValueError(f'Table ID {table_ID} not found!')
        suffixed_field_IDs = pd.read_table(tab_file, nrows=0).columns
        cache[table_ID] = suffixed_field_IDs
    return suffixed_field_IDs


def get_phenotype_from_tab_file(phenotype_name, all_phenotype_columns=False,
                                dtype=None):
    phenotype_table = get_phenotype_table()
    table_for_phenotype = phenotype_table.loc[[phenotype_name]]
    assert len(table_for_phenotype) == 1
    table_ID, field_ID = map(int, table_for_phenotype.iloc[0])
    if table_ID == 9797:
        tab_file = '/oak/stanford/groups/mrivas/ukbb/24983/' \
                   'phenotypedata/9796/9797/download/ukb9797.tab'
    else:
        raise ValueError(f'Table ID {table_ID} not found!')
    if all_phenotype_columns:
        all_field_IDs = get_full_field_IDs(table_ID)
        phenotype_column_names = all_field_IDs[
            all_field_IDs.str.startswith(f'f.{field_ID}')].tolist()
    else:
        # Most of the time phenotypes are not arrays or if they are we only care
        # about the first element.  So just return that as an array
        phenotype_column_names = [f'f.{field_ID}.0.0']
    phenotype = pd.read_table(
        tab_file, usecols=['f.eid'] + phenotype_column_names,
        index_col='f.eid', dtype=dtype)
    # Some phenotypes have missing/NaN values in the tab file; fill them in here
    phenotype.fillna(-9, inplace=True)
    # Because of the NaNs, pandas probably read this in as a float, so
    # convert back to int.  Just check that all of the values actually are
    # close to integers before converting.
    for column_name in phenotype.columns:
        # noinspection PyUnresolvedReferences
        if np.issubdtype(phenotype[column_name].dtype, np.floating):
            phenotype_as_int = phenotype[column_name].astype(int)
            if np.allclose(phenotype[column_name], phenotype_as_int):
                phenotype[column_name] = phenotype_as_int
    if not all_phenotype_columns:
        phenotype = phenotype.squeeze()
    return phenotype


def write_phenotype_to_phe_file(phenotype, generated_phe_file):
    assert isinstance(phenotype, pd.Series)
    if np.array_equal(np.sort(pd.unique(phenotype)), (0, 1)):
        phenotype += 1
    # FID/IID/pheno
    phenotype = phenotype.reset_index(name='phenotype').iloc[:, [0, 0, 1]]
    phenotype.to_csv(generated_phe_file, na_rep=-9,
                     sep='\t', header=False, index=False)


def get_field_IDs(phenotype_name):
    phenotype_table = get_phenotype_table()
    return phenotype_table.loc[[phenotype_name]]['FieldID']


def get_phe_file(phenotype_name, negative_values_to_keep=None):
    try:
        field_IDs = get_field_IDs(phenotype_name)
    except KeyError:
        # Special cases
        if phenotype_name == 'Family history of diabetes':
            generated_phe_file = \
                'generated_phe_files/family_history_of_diabetes.phe'
            if not os.path.exists(generated_phe_file):
                # https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=1010
                diabetes_code = 9
                none_of_above = [-17]  # none of the above
                # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20107
                father = get_phenotype('Illnesses of father',
                                       negative_values_to_keep=none_of_above)
                # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20110
                mother = get_phenotype('Illnesses of mother',
                                       negative_values_to_keep=none_of_above)
                # https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=20111
                siblings = get_phenotype('Illnesses of siblings',
                                       negative_values_to_keep=none_of_above)
                print(f'Father: {father.value_counts()}')
                print(f'Mother: {mother.value_counts()}')
                print(f'Siblings: {siblings.value_counts()}')
                phenotype = ((father == diabetes_code) |
                             (mother == diabetes_code) |
                             (siblings == diabetes_code)).astype(int) + 1
                print(f'Value counts: {phenotype.value_counts()}')
                # noinspection PyUnresolvedReferences
                assert not phenotype.isnull().any()
                write_phenotype_to_phe_file(phenotype, generated_phe_file)
            return generated_phe_file, None
        elif phenotype_name in ('type_1_diabetes', 'type_2_diabetes'):
            generated_phe_files = {
                phenotype_name: f'generated_phe_files/{phenotype_name}.phe'
                for phenotype_name in ('type_1_diabetes', 'type_2_diabetes')}
            if not all(map(os.path.exists, generated_phe_files.values())):
                eastwood = pd.read_csv('/oak/stanford/groups/mrivas/'
                                       'users/nasa/eastwood_output.csv',
                                       index_col='n_eid')
                A1C = pd.read_table(
                    '/oak/stanford/groups/mrivas/projects/biomarkers/'
                    'covariate_corrected/phenotypes/raw/biomarkers.phe',
                    usecols=['f.eid', 'f.30750.0.0'], index_col='f.eid',
                    squeeze=True)
                # Exclude T1D/gestational
                eastwood = eastwood[
                    (eastwood.sr_prob_t1_diabetes ==
                     'Type 1 diabetes unlikely') &
                    (eastwood.sr_poss_t1_diabetes ==
                     'Type 1 diabetes unlikely') &
                    (eastwood.sr_poss_gest_diabetes ==
                     'Gestational diabetes unlikely/impossible')
                ]
                # Cases are probable or possible T2D; the rest are controls
                T2D = (eastwood.sr_prob_t2_diabetes ==
                       'Probable type 2 diabetes') | (
                    eastwood.sr_poss_t2_diabetes == 'Possible type 2 diabetes')
                T2D = T2D.astype(int)
                # Exclude controls with A1C >= 39
                # (undiagnosed prediabetes/diabetes?)
                T2D = T2D[~((T2D == 0) & (A1C.reindex(T2D.index) >= 39))]
                # Save
                write_phenotype_to_phe_file(
                    T2D, generated_phe_files['type_2_diabetes'])
            return generated_phe_files[phenotype_name], None
        else:
            raise ValueError(f'Phenotype {phenotype_name} not found!')
    # Note: using .loc[[phenotype_name]] instead of .loc[phenotype_name]
    # ensure that field_IDs is a Series even if there is only one field ID
    for field_ID in field_IDs:
        # noinspection PyUnresolvedReferences
        try:
            phenotype_file = subprocess.check_output(
                # can't be any #s before/after field_ID; skip HC
                f'grep [^0-9C]{field_ID}.phe '
                f'/oak/stanford/groups/mrivas/ukbb/master_phe/phe_files.lst',
                shell=True).rstrip().decode()
            if '\n' in phenotype_file:
                phenotype_file = phenotype_file.split('\n')[0]
            phenotype_file = phenotype_file.replace(
                'phenotypedata', 'phenotypedata/old')
            return phenotype_file, field_ID
        except subprocess.CalledProcessError:
            # The phe file for field_ID has not been created yet
            continue
    else:
        # Generate the phe file directly from the tab file
        phe_files_dir = 'generated_phe_files'
        os.makedirs(phe_files_dir, exist_ok=True)
        generated_phe_file = \
            f'{phe_files_dir}/{phenotype_name.replace("/", "_")}.phe'
        if not os.path.exists(generated_phe_file):
            print(f'Generating phe file for {phenotype_name}...')
            phenotype = get_phenotype_from_tab_file(phenotype_name)
            # noinspection PyUnresolvedReferences
            was_originally_integer = np.issubdtype(phenotype.dtype, np.integer)
            # Set negatives to missing, except for the negative values in
            # negative_values_to_keep which we'll keep as-is
            if negative_values_to_keep is None:
                print('WARNING: setting -ve values to missing')
                phenotype[phenotype < 0] = -9
            elif not negative_values_to_keep == 'all':
                print(f'WARNING: setting -ve values except '
                      f'{negative_values_to_keep} to missing')
                phenotype[(phenotype < 0) & ~(phenotype.isin(
                    negative_values_to_keep))] = -9
            print(f'Value counts: {phenotype.value_counts()}')
            assert not phenotype.isnull().any()
            # noinspection PyUnresolvedReferences
            assert np.issubdtype(phenotype.dtype, np.integer) == \
                   was_originally_integer
            print(f'Saving to {generated_phe_file}...')
            write_phenotype_to_phe_file(phenotype, generated_phe_file)
            print('Done!')
        return generated_phe_file, None


def get_phenotype(phenotype_name, remove_negative_nines=True,
                  negative_values_to_keep=None):
    if phenotype_name == 'diabetes_adj_BMI':
        phenotype = get_phenotype(
            'diabetes', remove_negative_nines=remove_negative_nines)
        phenotype.name = 'diabetes_adj_BMI'
        return phenotype
    if phenotype_name == 'Obesity':
        BMI = get_phenotype('Body mass index (BMI)',
                            remove_negative_nines=remove_negative_nines)
        # noinspection PyTypeChecker
        obesity = (BMI >= 30).astype(int)
        obesity.is_binary = True
        if not remove_negative_nines:
            obesity[BMI == -9] = -9
        obesity.name = 'Obesity'
        return obesity
    phenotype_file, field_ID = get_phe_file(
        phenotype_name, negative_values_to_keep)
    phenotype = pd.read_table(phenotype_file, header=None,
                              delim_whitespace=True,
                              usecols=[0, 2], index_col=0, squeeze=True)
    # Filter to unrelated white British
    filtered_indivs_file = '/oak/stanford/groups/mrivas/ukbb24983/sqc/' \
                           'population_stratification/' \
                           'ukb24983_white_british.phe'
    filtered_indivs = pd.read_table(
        filtered_indivs_file, header=None, usecols=[0], squeeze=True)
    phenotype = phenotype[phenotype.index.isin(filtered_indivs)]
    assert not (phenotype == -9).all()
    if remove_negative_nines:
        phenotype = phenotype[phenotype != -9]
    phenotype.is_binary = len(np.unique(phenotype)) == 2
    if phenotype.is_binary:
        phenotype[phenotype != -9] -= 1  # convert to 1 = case, 0 = control
        phenotype = phenotype.astype(int)
        phenotype.is_binary = True  # new array, need to set again
    phenotype.name = phenotype_name
    return phenotype
