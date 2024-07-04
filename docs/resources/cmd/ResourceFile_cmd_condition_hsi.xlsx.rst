cmd condition hsi (.xlsx)
=========================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/cmd/ResourceFile_cmd_condition_hsi.xlsx>`

.. contents::

info
----

====  ================================================================================================================================================================================================================================================================
  ..  ResourceFile\_cmd\_condition\_hsi
====  ================================================================================================================================================================================================================================================================
   0  Created by Britta Jewell on 14 April 2021
   1
   2  This file reads in parameters for the HSI events (diagnostics and treatment) of each condition annually in the CMD module in the TLO model. Each tab represents one condition modelled. Where possible, sources are listed to the right of the parameter values.
====  ================================================================================================================================================================================================================================================================

diabetes
--------

====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================
  ..  parameter\_name                                          value  Unnamed: 2    Unnamed: 3
====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================
   0  test\_item\_code                                      216                     Blood glucose level test
   1  sensitivity\_of\_assessment                             1
   2  medication\_item\_code                                233                     First line treatment (Metformin hydrochloride 500mg\_100\_CMST)
   3  pr\_assessed\_other\_symptoms                           0.1
   4  pr\_treatment\_works                                    0.42                  Prospective Diabetes Study (UKPDS) Group: Effect of intensive blood glucose control with metformin on complications in overweight patients with type 2 diabetes (UKPDS 34). Lancet. 1998, 352 (9131): 854-865.
   5  pr\_diagnosed                                           0.3846                Price et al. Prevalence of obesity, hypertension, and diabetes, and cascade of care in sub-Saharan Africa: a cross-sectional, population-based study in rural and urban Malawi. The Lancet Diabetes and Endocrinology 2018:6 208-22.
   6  pr\_seeking\_further\_appt\_if\_drug\_not\_available    0.8
====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================

hypertension
------------

====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================
  ..  parameter\_name                                          value  Unnamed: 2    Unnamed: 3
====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================
   0  sensitivity\_of\_assessment                             1
   1  medication\_item\_code                                221                     First line treatment (Hydrochlorothiazide 25mg\_1000\_CMST)
   2  pr\_assessed\_other\_symptoms                           0.05
   3  pr\_treatment\_works                                    0
   4  pr\_diagnosed                                           0.3797                Price et al. Prevalence of obesity, hypertension, and diabetes, and cascade of care in sub-Saharan Africa: a cross-sectional, population-based study in rural and urban Malawi. The Lancet Diabetes and Endocrinology 2018:6 208-22.
   5  pr\_seeking\_further\_appt\_if\_drug\_not\_available    0.8
====  ====================================================  ========  ============  ====================================================================================================================================================================================================================================

chronic_lower_back_pain
-----------------------

====  ====================================================  =======  ============  =========================
  ..  parameter\_name                                         value  Unnamed: 2    Unnamed: 3
====  ====================================================  =======  ============  =========================
   0  sensitivity\_of\_assessment                               1
   1  medication\_item\_code                                  226                  Aspirin 300mg\_1000\_CMST
   2  pr\_assessed\_other\_symptoms                             0
   3  pr\_treatment\_works                                      0
   4  pr\_diagnosed                                             0
   5  pr\_seeking\_further\_appt\_if\_drug\_not\_available      0.1
====  ====================================================  =======  ============  =========================

chronic_kidney_disease
----------------------

====  ====================================================  =======  ============  =============================================================================
  ..  parameter\_name                                         value  Unnamed: 2    Unnamed: 3
====  ====================================================  =======  ============  =============================================================================
   0  test\_item\_code                                        47
   1  pr\_assessed\_in\_generic\_appt\_level1                  1
   2  medication\_item\_code                                2064                   Dianeal + Dextrose 1.5% intraperitoneal dialysis soln. with minicap\_2L\_CMST
   3  pr\_assessed\_other\_symptoms                            0
   4  pr\_treatment\_works                                     0
   5  pr\_diagnosed                                            0.02
   6  pr\_seeking\_further\_appt\_if\_drug\_not\_available     0.8
====  ====================================================  =======  ============  =============================================================================

chronic_ischemic_hd
-------------------

====  ====================================================  =======  ============  =========================
  ..  parameter\_name                                         value  Unnamed: 2    Unnamed: 3
====  ====================================================  =======  ============  =========================
   0  sensitivity\_of\_assessment                              1
   1  medication\_item\_code                                 226                   Aspirin 300mg\_1000\_CMST
   2  pr\_assessed\_other\_symptoms                            0
   3  pr\_treatment\_works                                     0
   4  pr\_diagnosed                                            0.05
   5  pr\_seeking\_further\_appt\_if\_drug\_not\_available     0.8
====  ====================================================  =======  ============  =========================

