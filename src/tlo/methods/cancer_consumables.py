"""
This file stores defines the consumables required within the cancer modules
"""
from typing import Dict

from tlo import Module


def get_consumable_item_codes_cancers(cancer_module: Module) -> Dict[str, int]:
    """
    Returns dict the relevant item_codes for the consumables across the five cancer modules. This is intended to prevent
    repetition within module code.
    """

    def get_list_of_items(item_list):
        item_lookup_fn = cancer_module.sim.modules['HealthSystem'].get_item_code_from_item_name
        return list(map(item_lookup_fn, item_list))

    cons_dict = dict()

    # Add items that are needed for all cancer modules
    # todo: @Eva - add syringes, dressing
    cons_dict['screening_biopsy_core'] = get_list_of_items(['Biopsy needle'])

    cons_dict['screening_biopsy_optional'] = \
        get_list_of_items(['Specimen container',
                           'Lidocaine, injection, 1 % in 20 ml vial',
                           'Gauze, absorbent 90cm x 40m_each_CMST',
                           'Disposables gloves, powder free, 100 pieces per box'])

    cons_dict['treatment_surgery_core'] = \
        get_list_of_items(['Halothane (fluothane)_250ml_CMST',
                           'Scalpel blade size 22 (individually wrapped)_100_CMST'])

    cons_dict['treatment_surgery_optional'] = \
        get_list_of_items(['Sodium chloride, injectable solution, 0,9 %, 500 ml',
                           'Paracetamol, tablet, 500 mg',
                           'Pethidine, 50 mg/ml, 2 ml ampoule',
                           'Suture pack',
                           'Gauze, absorbent 90cm x 40m_each_CMST',
                           'Cannula iv  (winged with injection pot) 18_each_CMST'])

    cons_dict['palliation'] = \
        get_list_of_items(['morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA',
                           'Diazepam, injection, 5 mg/ml, in 2 ml ampoule',
                           # N.B. This is not an exhaustive list of drugs required for palliation
                           ])

    cons_dict['iv_drug_cons'] = \
        get_list_of_items(['Cannula iv  (winged with injection pot) 18_each_CMST',
                           'Giving set iv administration + needle 15 drops/ml_each_CMST',
                           'Disposables gloves, powder free, 100 pieces per box'
                           ])

    # Add items that are specific to each cancer module
    if 'BreastCancer' == cancer_module.name:

        # TODO: @Eva chemotharpy protocols??: TAC(Taxotere, Adriamycin, and Cyclophosphamide), AC (anthracycline and
        #  cyclophosphamide) +/-Taxane, TC (Taxotere and cyclophosphamide), CMF (cyclophosphamide, methotrexate,
        #  and fluorouracil), FEC-75 (5-Fluorouracil, Epirubicin, Cyclophosphamide). HER 2 +: Add Trastuzumab

        # only chemotherapy i consumable list which is also in suggested protocol is cyclo
        cons_dict['treatment_chemotherapy'] = get_list_of_items(['Cyclophosphamide, 1 g'])

    elif 'ProstateCancer' == cancer_module.name:

        # TODO: @Eva Prostate specific antigen test is listed in ResourceFile_Consumables_availability_and_usage but not
        #  ResourceFile_Consumables_Items_and_Package
        # cons_dict['screening_psa_test_core'] = get_list_of_items(['Prostate specific antigen test'])

        cons_dict['screening_psa_test_optional'] = \
            get_list_of_items(['Blood collecting tube, 5 ml',
                               'Disposables gloves, powder free, 100 pieces per box'])

    elif 'BladderCancer' == cancer_module.name:
        # Note: bladder cancer is not in the malawi STG 2023 therefore no details on chemotherapy

        # TODO: @Eva cytoscope is listed in ResourceFile_Consumables_availability_and_usage but not
        #  ResourceFile_Consumables_Items_and_Packages
        # cons_dict['screening_cystoscopy_core'] = get_list_of_items(['Cytoscope'])

        cons_dict['screening_cystoscope_optional'] = get_list_of_items(['Specimen container'])

    elif 'OesophagealCancer' == cancer_module.name:

        # TODO: @Eva endoscope is listed in ResourceFile_Consumables_availability_and_usage but not
        #  ResourceFile_Consumables_Items_and_Packages
        # cons_dict['screening_endoscope_core'] = get_list_of_items(['Endoscope'])

        cons_dict['screening_endoscope_optional'] =\
            get_list_of_items(['Specimen container',
                               'Gauze, absorbent 90cm x 40m_each_CMST'])

        cons_dict['treatment_chemotherapy'] = get_list_of_items(['Cisplatin 50mg Injection'])

    return cons_dict
