"""
This file stores defines the consumables required within the cancer modules
"""

import numpy as np
import pandas as pd

from tlo import logging


def get_consumable_item_codes_cancers(self, cons_dict):
    """
    This function stores the relevant item codes for cancer consumables across the five cancer modules to prevent
    repetition within module code.
    """

    def get_list_of_items(item_list):
        item_code_function = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        codes = [item_code_function(item) for item in item_list]
        return codes

    # to do: add syringes, dressing
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

    # This is not an exhaustive list of drugs required for palliation
    cons_dict['palliation'] = \
        get_list_of_items(['morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA',
                           'Diazepam, injection, 5 mg/ml, in 2 ml ampoule',
                           ])

    cons_dict['iv_drug_cons'] = \
        get_list_of_items(['Cannula iv  (winged with injection pot) 18_each_CMST',
                           'Giving set iv administration + needle 15 drops/ml_each_CMST',
                           'Disposables gloves, powder free, 100 pieces per box'
                           ])

    if self == self.sim.modules['BreastCancer']:

        # TODO: chemotharpy protocols??: TAC(Taxotere, Adriamycin, and Cyclophosphamide), AC (anthracycline and
        #  cyclophosphamide) +/-Taxane, TC (Taxotere and cyclophosphamide), CMF (cyclophosphamide, methotrexate,
        #  and fluorouracil), FEC-75 (5-Fluorouracil, Epirubicin, Cyclophosphamide). HER 2 +: Add Trastuzumab

        # only chemotherapy i consumable list which is also in suggested protocol is cyclo
        cons_dict['treatment_chemotherapy'] = get_list_of_items(['Cyclophosphamide, 1 g'])

    elif self == self.sim.modules['ProstateCancer']:

        cons_dict['screening_psa_test_core'] = get_list_of_items(['Prostate specific antigen test'])

        cons_dict['screening_psa_test_optional'] = \
            get_list_of_items(['Blood collecting tube, 5 ml'
                               'Disposables gloves, powder free, 100 pieces per box'])

    elif self == self.sim.modules['Bladder_cancer']:
        # Note: bladder cancer is not in the malawi STG 2023 therefore no details on chemotherapy

        cons_dict['screening_cytoscopy_core'] = get_list_of_items(['Cytoscope'])

        cons_dict['screening_cytoscope_optional'] = get_list_of_items(['Specimen container'])

    elif self == self.sim.modules['OesophagealCancer']:

        cons_dict['screening_endoscope_core'] = get_list_of_items(['Endoscope'])

        cons_dict['screening_endoscope_optional'] =\
            get_list_of_items(['Specimen container',
                               'Gauze, absorbent 90cm x 40m_each_CMST'])

        cons_dict['treatment_chemotherapy'] = get_list_of_items(['Cisplatin 50mg Injection'])

