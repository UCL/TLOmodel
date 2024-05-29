"""
This file stores defines the consumables required within the cancer modules
"""
from typing import Dict

from tlo import Module


def get_consumable_item_codes_cancers(self, cancer_module: Module) -> Dict[str, int]:
    """
    Returns dict the relevant item_codes for the consumables across the five cancer modules. This is intended to prevent
    repetition within module code.
    """

    get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

    cons_dict = dict()

    # Add items that are needed for all cancer modules
    cons_dict['screening_biopsy_core'] = \
        {get_item_code("Biopsy needle"): 1}

    cons_dict['screening_biopsy_optional'] = \
        {get_item_code("Specimen container"): 1,
         get_item_code("Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml"): 1,
         get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 30,
         get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
         get_item_code("Syringe, needle + swab"): 1}

    cons_dict['treatment_surgery_core'] = \
        {get_item_code("Halothane (fluothane)_250ml_CMST"): 100,
         get_item_code("Scalpel blade size 22 (individually wrapped)_100_CMST"): 1}

    cons_dict['treatment_surgery_optional'] = \
        {get_item_code("Sodium chloride, injectable solution, 0,9 %, 500 ml"): 2000,
         get_item_code("Paracetamol, tablet, 500 mg"): 8000,
         get_item_code("Pethidine, 50 mg/ml, 2 ml ampoule"): 6,
         get_item_code("Suture pack"): 1,
         get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 30,
         get_item_code("Cannula iv  (winged with injection pot) 18_each_CMST"): 1}

    cons_dict['palliation'] = \
        {get_item_code("morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA"): 1,
         get_item_code("Diazepam, injection, 5 mg/ml, in 2 ml ampoule"): 3,
         get_item_code("Syringe, needle + swab"): 4}
    # N.B. This is not an exhaustive list of drugs required for palliation

    cons_dict['treatment_chemotherapy_core'] = \
        {get_item_code("Cyclophosphamide, 1 g"): 16800}

    cons_dict['iv_drug_cons'] = \
        {get_item_code("Cannula iv  (winged with injection pot) 18_each_CMST"): 1,
         get_item_code("Giving set iv administration + needle 15 drops/ml_each_CMST"): 1,
         get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
         get_item_code("Gauze, swabs 8-ply 10cm x 10cm_100_FF010800_CMST"): 84}

    # Add items that are specific to a particular cancer module
    if 'ProstateCancer' == cancer_module.name:

        # TODO: @Sakshi the script to create RF_Consumables_Items_and_Pkgs needs to be re-run
        cons_dict['screening_psa_test_core'] = \
            {get_item_code("Prostate specific antigen test"): 1}

        cons_dict['screening_psa_test_optional'] = \
            {get_item_code("Blood collecting tube, 5 ml"): 1,
             get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
             get_item_code("Gauze, swabs 8-ply 10cm x 10cm_100_FF010800_CMST"): 1}

    elif 'BladderCancer' == cancer_module.name:
        # Note: bladder cancer is not in the malawi STG 2023 therefore no details on chemotherapy

        # TODO: @Sakshi the script to create RF_Consumables_Items_and_Pkgs needs to be re-run
        cons_dict['screening_cystoscopy_core'] = \
            {get_item_code("Cystoscope"): 1}

        cons_dict['screening_cystoscope_optional'] = \
            {get_item_code("Specimen container"): 1,
             get_item_code("Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml"): 1,
             get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 30,
             get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
             get_item_code("Syringe, needle + swab"): 1}

    elif 'OesophagealCancer' == cancer_module.name:

        # TODO: @Sakshi the script to create RF_Consumables_Items_and_Pkgs needs to be re-run
        cons_dict['screening_endoscope_core'] = \
            {get_item_code("Endoscope"): 1}

        cons_dict['screening_endoscope_optional'] = \
            {get_item_code("Specimen container"): 1,
             get_item_code("Gauze, absorbent 90cm x 40m_each_CMST"): 30,
             get_item_code("Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml"): 1,
             get_item_code("Disposables gloves, powder free, 100 pieces per box"): 1,
             get_item_code("Syringe, needle + swab"): 1}

    return cons_dict
