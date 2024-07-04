PostnatalSupervisor (.xlsx)
===========================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_PostnatalSupervisor.xlsx>`

.. contents::

parameter_values
----------------

====  ==================================================  ==========================================================  ==========================================================================================================================================================================================================================================
  ..  parameter\_name                                     value                                                       Unnamed: 2
====  ==================================================  ==========================================================  ==========================================================================================================================================================================================================================================
   0  prob\_obstetric\_fistula                            [0.00426, 0.00426]
   1  rr\_obstetric\_fistula\_obstructed\_labour          [14.8, 14.8]                                                https://pubmed.ncbi.nlm.nih.gov/28550462/
   2  prevalence\_type\_of\_fistula                       [[0.924, 0.076] , [0.924, 0.076]]                           https://pubmed.ncbi.nlm.nih.gov/17869256/#:~:text=This%20study%20from%20Southern%20Malawi,were%20not%20of%20obstetric%20origin.
   3
   4  prob\_secondary\_pph                                [0.0002, 0.00034]
   5  rr\_secondary\_pph\_endometritis                    [1, 1]                                                      Delete?
   6  prob\_secondary\_pph\_severity                      [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]
   7  cfr\_secondary\_postpartum\_haemorrhage             [0.22, 0.08]
   8
   9  prob\_htn\_resolves                                 [0.167, 0.167]
  10  weekly\_prob\_gest\_htn\_pn                         [0.0025, 0.0025]
  11  rr\_gest\_htn\_obesity                              [3.31, 3.31]
  12  weekly\_prob\_pre\_eclampsia\_pn                    [0.0017, 0.0017]                                            [0.0016, 0.0016]
  13  rr\_pre\_eclampsia\_obesity                         [3.2, 3.2]
  14  rr\_pre\_eclampsia\_chronic\_htn                    [2.26, 2.26]
  15  rr\_pre\_eclampsia\_diabetes\_mellitus              [3.7, 3.7]
  16  probs\_for\_mgh\_matrix\_pn                         [[0.918, 0.032, 0.05, 0, 0], [0.918, 0.032, 0.05, 0, 0]]    [[0.884, 0.066, 0.05, 0, 0], [0.884, 0.066, 0.05, 0, 0]]
  17  probs\_for\_sgh\_matrix\_pn                         [[0, 0.92, 0, 0.08, 0], [0, 0.92, 0, 0.08, 0]]
  18  probs\_for\_mpe\_matrix\_pn                         [[0, 0, 0.95, 0.05, 0], [0, 0, 0.95, 0.05, 0]]              [[0, 0, 0.89, 0.11, 0], [0, 0, 0.89, 0.11, 0]]
  19  probs\_for\_spe\_matrix\_pn                         [[0, 0, 0, 0.95, 0.05], [0, 0, 0, 0.95, 0.05]]
  20  probs\_for\_ec\_matrix\_pn                          [[0, 0, 0, 0, 1],  [0, 0, 0, 0, 1]]
  21  cfr\_eclampsia                                      [0.028, 0.03]
  22  cfr\_severe\_pre\_eclampsia                         [0.018, 0.018]
  23  weekly\_prob\_death\_severe\_gest\_htn              [0.00002, 0.00002]
  24
  25  baseline\_prob\_anaemia\_per\_week                  [0.017, 0.024]
  26  rr\_anaemia\_maternal\_malaria                      [1.45, 1.45]
  27  rr\_anaemia\_recent\_haemorrhage                    [1.93, 1.93]
  28  rr\_anaemia\_hiv\_no\_art                           [4.19, 4.19]
  29  prob\_type\_of\_anaemia\_pn                         [[0.52,0.475, 0.005], [0.52,0.475, 0.005]]
  30
  31  prob\_late\_sepsis\_endometritis                    [0.000069, 0.000062]
  32  rr\_sepsis\_endometritis\_post\_cs                  [12.1, 12.1]
  33  prob\_late\_sepsis\_urinary\_tract                  [0.000054, 0.000054]
  34  prob\_late\_sepsis\_skin\_soft\_tissue              [0.000039, 0.000035]
  35  rr\_sepsis\_sst\_post\_cs                           [3.9, 3.9]                                                  https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-018-1891-1
  36  cfr\_postpartum\_sepsis                             [0.75, 0.49]
  37
  38  prob\_early\_onset\_neonatal\_sepsis\_week\_1       [0.02, 0.017]
  39  rr\_eons\_maternal\_chorio                          [6.6, 6.6]
  40  rr\_eons\_maternal\_prom                            [4.9, 4.9]
  41  rr\_eons\_preterm\_neonate                          [3.36, 3.36]
  42  cfr\_early\_onset\_neonatal\_sepsis                 [0.064, 0.056]
  43  prob\_late\_onset\_neonatal\_sepsis                 [0.0045, 0.0038]
  44  cfr\_late\_neonatal\_sepsis                         [0.064, 0.056]
  45  prob\_sepsis\_disabilities                          [[0.4, 0.1,  0.3,  0.1, 0.1], [0.4, 0.1,  0.3,  0.1, 0.1]]
  46
  47  prob\_care\_seeking\_postnatal\_emergency           [0.782, 0.782]                                              Chinkhumba, J. et al. (2017) ‘Household costs and time to seek care for pregnancy related complications: The role of results-based financing’, PLOS ONE. Public Library of Science, 12(9), p. e0182326. doi: 10.1371/JOURNAL.PONE.0182326.
  48  prob\_care\_seeking\_postnatal\_emergency\_neonate  [0.782, 0.782]                                              Chinkhumba, J. et al. (2017) ‘Household costs and time to seek care for pregnancy related complications: The role of results-based financing’, PLOS ONE. Public Library of Science, 12(9), p. e0182326. doi: 10.1371/JOURNAL.PONE.0182326.
  49  odds\_care\_seeking\_fistula\_repair                [1.5, 1.5]
  50  aor\_cs\_fistula\_age\_15\_19                       [0.31, 0.31]
  51  aor\_cs\_fistula\_age\_lowest\_education            [0.69, 0.69]
  52
  53  treatment\_effect\_iron\_folic\_acid\_anaemia       [0.3, 0.3]
  54  treatment\_effect\_early\_init\_bf                  [0.55, 0.55]
  55  treatment\_effect\_abx\_prom                        [0.67, 0.67]
  56  treatment\_effect\_cord\_care                       [0.77,0.77]
  57  treatment\_effect\_clean\_birth                     [0.73, 0.73]
  58  treatment\_effect\_anti\_htns\_progression\_pn      [0.49, 0.49]
====  ==================================================  ==========================================================  ==========================================================================================================================================================================================================================================

parameter_values_old
--------------------

====  ======================================================  ===========================  ============  ===============================================================================================================================================================
  ..  parameter\_name                                         value                        Unnamed: 2    Source
====  ======================================================  ===========================  ============  ===============================================================================================================================================================
   0  prob\_htn\_resolves                                     0.8                                        Dummy value
   1  prob\_secondary\_pph                                    0.05                                       Dummy value
   2  cfr\_secondary\_pph                                     0.0014                                     Dummy value
   3  cfr\_postnatal\_sepsis                                  0.0014                                     Dummy value
   4  prob\_secondary\_pph\_severity                          [0.33, 0.33, 0.34]                         Dummy value
   5  prob\_obstetric\_fistula                                0.1                                        Dummy value
   6  prevalence\_type\_of\_fistula                           [0.5, 0.5]                                 Dummy value
   7  prob\_iron\_def\_per\_week\_pn                          0.01                                       Dummy value
   8  rr\_iron\_def\_ifa\_pn                                  0.43                                       Daily oral iron supplementation during pregnancy - https://pubmed.ncbi.nlm.nih.gov/26198451/
   9  prob\_folate\_def\_per\_week\_pn                        0.0025                                     Dummy value
  10  rr\_folate\_def\_ifa\_pn                                0.43                                       Daily oral iron supplementation during pregnancy - https://pubmed.ncbi.nlm.nih.gov/26198451/
  11  prob\_b12\_def\_per\_week\_pn                           0.0025                                     Dummy value
  12  baseline\_prob\_anaemia\_per\_week                      0.001                                      Dummy value
  13  prob\_type\_of\_anaemia\_pn                             [0.33, 0.33, 0.34]                         Dummy value
  14  rr\_anaemia\_if\_iron\_deficient\_pn                    1.5                                        Dummy value
  15  rr\_anaemia\_if\_folate\_deficient\_pn                  1.25                                       Dummy value
  16  rr\_anaemia\_if\_b12\_deficient\_pn                     1.25                                       Dummy value
  17  prob\_endometritis\_pn                                  0.1                                        Dummy value
  18  prob\_urinary\_tract\_inf\_pn                           0.1                                        Dummy value
  19  prob\_skin\_soft\_tissue\_inf\_pn                       0.1                                        Dummy value
  20  prob\_other\_inf\_pn                                    0.1                                        Dummy value
  21  treatment\_effect\_early\_init\_bf                      0.85                                       Breastfeeding effect sizes on mortality in LiST (technical note found in spectrum help section)
  22  treatment\_effect\_abx\_prom                            0.61                                       Antibiotics for pre-term pre-labour rupture of membranes: prevention of neonatal deaths due to complications of pre-term birth and infection
  23  treatment\_effect\_inj\_abx\_sep                        0.35                                       Effect of case management on neonatal mortality due to sepsis and pneumonia (2011) https://pubmed.ncbi.nlm.nih.gov/21501430/
  24  treatment\_effect\_supp\_care\_sep                      0.2                                        Effect of case management on neonatal mortality due to sepsis and pneumonia (2011) https://pubmed.ncbi.nlm.nih.gov/21501430/
  25  treatment\_effect\_cord\_care                           0.77                                       Clean birth and postnatal care practices to reduce neonatal deaths from sepsis and tetanus: a systematic review and Delphi estimation of mortality effect
  26  treatment\_effect\_clean\_birth                         0.73                                       Clean birth and postnatal care practices to reduce neonatal deaths from sepsis and tetanus: a systematic review and Delphi estimation of mortality effect
  27  prob\_early\_onset\_neonatal\_sepsis\_week\_1           0.25                                       Dummy value
  28  cfr\_early\_onset\_neonatal\_sepsis                     0.25                                       Dummy value
  29  prob\_late\_sepsis\_endometritis                        0.1                                        Dummy value
  30  prob\_late\_sepsis\_urinary\_tract\_inf                 0.1                                        Dummy value
  31  prob\_late\_sepsis\_skin\_soft\_tissue\_inf             0.1                                        Dummy value
  32  prob\_late\_sepsis\_other\_maternal\_infection\_pp      0.1                                        Dummy value
  33  prob\_late\_onset\_neonatal\_sepsis                     0.1                                        Dummy value
  34  cfr\_late\_neonatal\_sepsis                             0.25                                       Dummy value
  35  prob\_sepsis\_disabilities                              [0.4, 0.1,  0.3,  0.1, 0.1]                Dummy value
  36  prob\_htn\_persists                                     0.2                                        Dummy value
  37  weekly\_prob\_gest\_htn\_pn                             0.001                                      Dummy value
  38  weekly\_prob\_pre\_eclampsia\_pn                        0.001                                      Dummy value
  39  probs\_for\_mgh\_matrix\_pn                             [0.8, 0.1, 0.1, 0.0, 0.0]
  40  probs\_for\_sgh\_matrix\_pn                             [0.0, 0.8, 0.0, 0.2, 0.0]
  41  probs\_for\_mpe\_matrix\_pn                             [0.0, 0.0, 0.8, 0.2, 0.0]
  42  probs\_for\_spe\_matrix\_pn                             [0.0, 0.0, 0.0, 0.6, 0.4]
  43  probs\_for\_ec\_matrix\_pn                              [0.0, 0.0, 0.0, 0.0, 1]
  44  cfr\_eclampsia\_pn                                      0.0014                       0.15          Dummy value
  45  cfr\_severe\_htn\_pn                                    0.0014
  46  prob\_attend\_pnc2                                      0.25                                       Dummy value
  47  prob\_attend\_pnc3                                      0.25                                       Dummy value
  48  treatment\_effect\_anti\_htns\_progression\_pn          0.49                                       Antihypertensive drug therapy for mild to moderate hypertension during pregnancy - https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD002252.pub4/full
  49  treatment\_effect\_parenteral\_antibiotics              0.2                                        Estimating the impact of interventions on causespecific maternal mortality: a Delphi approach
  50  treatment\_effect\_bemonc\_care\_pph                    0.25                                       Estimating the impact of interventions on causespecific maternal mortality: a Delphi approach
  51  treatment\_effect\_anti\_htns                           0.5                                        Estimating the impact of interventions on causespecific maternal mortality: a Delphi approach
  52  treatment\_effect\_mag\_sulph                           0.4                                        Estimating the impact of interventions on causespecific maternal mortality: a Delphi approach
  53  neonatal\_sepsis\_treatment\_effect                     0.2                                        Effect of case management on neonatal mortality due to sepsis and pneumonia (2011) https://pubmed.ncbi.nlm.nih.gov/21501430/
  54  severity\_late\_infection\_pn                           [0.64, 0.22, 0.14]                         Dummy value
  55  prob\_care\_seeking\_postnatal\_emergency               0.5                                        Dummy value
  56  prob\_care\_seeking\_postnatal\_emergency\_neonate      0.5                                        Dummy value
  57  odds\_care\_seeking\_fistula\_repair                    1.5                                        Treatment-seeking for vaginal fistula in sub-Saharan Africa - this value is made up
  58  aor\_cs\_fistula\_age\_15\_19                           0.31                                       Treatment-seeking for vaginal fistula in sub-Saharan Africa
  59  aor\_cs\_fistula\_age\_lowest\_education                0.69                                       Treatment-seeking for vaginal fistula in sub-Saharan Africa
  60  prob\_pnc1\_at\_day\_7                                  0.4                                        Dummy value
  61  multiplier\_for\_care\_seeking\_with\_comps             2                                          Dummy value
  62  sensitivity\_bp\_monitoring\_pn                         0.9
  63  specificity\_bp\_monitoring\_pn                         0.9
  64  sensitivity\_urine\_protein\_1\_plus\_pn                0.9
  65  specificity\_urine\_protein\_1\_plus\_pn                0.9
  66  sensitivity\_poc\_hb\_test\_pn                          0.9
  67  specificity\_poc\_hb\_test\_pn                          0.9
  68  sensitivity\_maternal\_sepsis\_assessment               0.9
  69  sensitivity\_pph\_assessment                            0.9
  70  sensitivity\_lons\_assessment                           0.9
  71  sensitivity\_eons\_assessment                           0.9
  72  prob\_intervention\_delivered\_sep\_assessment\_pnc     0.9
  73  prob\_intervention\_delivered\_pph\_assessment\_pnc     0.9
  74  prob\_intervention\_delivered\_urine\_ds\_pnc           0.9
  75  prob\_intervention\_delivered\_bp\_pnc                  0.9
  76  prob\_intervention\_delivered\_hiv\_test\_pnc           0.9
  77  prob\_intervention\_poct\_pnc                           0.9
  78  prob\_intervention\_neonatal\_sepsis\_pnc               0.9
  79  prob\_intervention\_delivered\_depression\_screen\_pnc  0.9
  80                                                          0.9
====  ======================================================  ===========================  ============  ===============================================================================================================================================================

