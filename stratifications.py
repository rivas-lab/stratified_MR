import numpy as np
from itertools import product
# noinspection PyUnresolvedReferences
from pgenlib import PgenReader
from utils import pairwise


def stratify_by_tertiles(array):
    # noinspection PyUnresolvedReferences
    assert np.issubdtype(array.dtype, np.floating)
    # Return mask of tertiles
    tertiles = 0, 1/3, 2/3, 1
    tertile_names = 'low', 'medium', 'high'
    tertile_masks = {}
    for tertile_name, (low, high) in zip(
            tertile_names, pairwise(array.quantile(tertiles))):
        if tertile_name == 'high':
            mask = (array >= low) & (array <= high)
        else:
            mask = (array >= low) & (array < high)
        tertile_masks[tertile_name] = mask
    return tertile_masks


def stratify_binary(array):
    binary_masks = {'yes': array == 1, 'no': array == 0}
    assert (binary_masks['yes'] | binary_masks['no'] | np.isnan(array)).all()
    return binary_masks


def stratify_BMI(exposure):
    assert exposure.name == 'Body mass index (BMI)'
    BMI_range_masks = {
        'normal': (exposure < 25),
        'overweight': ((exposure >= 25) & (exposure < 30)),
        'obese': exposure >= 30
    }
    return BMI_range_masks


def stratify_BMI_fine_grained(exposure):
    assert exposure.name == 'Body mass index (BMI)'
    BMI_range_masks = {
        'normal': (exposure < 25),
        'overweight_low': ((exposure >= 25) & (exposure < 27.5)),
        'overweight_high': ((exposure >= 27.5) & (exposure < 30)),
        'obese_low': ((exposure >= 30) & (exposure < 35)),
        'obese_high': exposure >= 35,
    }
    return BMI_range_masks


def get_exposure_stratifications(exposure):
    assert exposure.name == 'Body mass index (BMI)'
    if hasattr(exposure, 'fine_grained_BMI'):
        exposure_masks = stratify_BMI_fine_grained(exposure)
    else:
        exposure_masks = stratify_BMI(exposure)
    return exposure_masks


def get_PRS_stratifications(PRS):
    return stratify_by_tertiles(PRS)


def get_family_history_stratifications(family_history):
    return stratify_binary(family_history)


def get_stratifications(exposure=None, PRS=None, family_history=None):
    # Stratify by exposure, PRS, and/or family history.
    # Returns a dictionary of {stratification name: mask of individuals}
    # 1. Get "marginal" strats for exposure, PRS and family history individually
    marginal_stratifications = {}
    if exposure is not None:
        marginal_stratifications['exposure'] = \
            get_exposure_stratifications(exposure)
    if PRS is not None:
        marginal_stratifications['PRS'] = get_PRS_stratifications(PRS)
    if family_history is not None:
        marginal_stratifications['family_history'] = \
            get_family_history_stratifications(family_history)
    # 2. Full strats are all possible combinations of marginal strats,
    # anded together
    stratifications = {}
    for stratification_names in product(*marginal_stratifications.values()):
        stratification_name = '_'.join(
            f'{stratification_name}_{array_name}'
            for array_name, stratification_name in zip(
                marginal_stratifications, stratification_names))
        masks = [marginal_stratifications[array_name][stratification_name]
                 for array_name, stratification_name in zip(
                marginal_stratifications, stratification_names)]
        stratification_mask = np.all(masks, axis=0)
        stratifications[stratification_name] = stratification_mask
    # 3. Print stats
    individuals_per_stratification = {
        stratification_name: stratification.sum()
        for stratification_name, stratification in stratifications.items()}
    print(f'Individuals in each stratification: '
          f'{individuals_per_stratification}')
    assert min(individuals_per_stratification.values()) > 50
    return stratifications
