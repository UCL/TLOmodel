Childhood Pneumonia (.xlsx)
===========================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Childhood_Pneumonia.xlsx>`

.. contents::

Cover Sheet
-----------

====  ============  =====================================  ===================
  ..  Unnamed: 0    Unnamed: 1                             Unnamed: 2
====  ============  =====================================  ===================
   0
   1                Name:                                  Childhood Pneumonia
   2
   3                Link to Python Code on GitHub:
   4                Link to Issue Discussion on GitHub:
   5
   6                Type of Module:                        Disease
   7
   8                Author:                                Ines
   9
  10                Major Amendments Revision History:
  11
  12                Other Aspects of the Model Addressed:
  13                Diseases:
  14                Interventions:
  15                Lifestyle Elements:
  16                P-variables:
  17                H-variables:
  18                E-variables:
  19
  20                Proccessed
  21                Completed by author:
  22                Coded in main framework:
  23                Reviwed by Team:
====  ============  =====================================  ===================

Structure
---------

====  ======================================================  ==============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ==================================================================================  ============  ============
  ..  Unnamed: 0                                              Unnamed: 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      Unnamed: 2                                                                          Unnamed: 3    Unnamed: 4
====  ======================================================  ==============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ==================================================================================  ============  ============
   0
   1
   2
   3
   4
   5
   6
   7
   8
   9
  10
  11
  12
  13
  14
  15
  16
  17
  18
  19
  20
  21
  22
  23
  24
  25
  26
  27
  28
  29
  30
  31
  32
  33
  34
  35
  36
  37
  38
  39
  40
  41
  42
  43  Description of Events and Parameters
  44
  45
  46
  47
  48  Parameter                                               Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Notes                                                                               Value         Ref
  49  base\_incidence\_pneumonia\_by pathogen[age\_category]  a list of incidence of diarrhoea caused by individual pathogens in each age group [0-11, 12-59]; pneumonia caused by RSV, rhinovirus, hMPV, parainfluenza, streptococcus, hib, TB, stapylococcus, influenza, and P. jirovecii in a susceptible population of children under 5, with base group of no handwashing with soap, indoor air pollution, and no exclusive or continued breastfeeding, HIV negative, no SAM, no pneumococcal vaccination, no hib vaccination, no influenza vaccination  base\_incidence\_pneumonia\_by\_RSV[at age 0-11, 12-59]
  50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_rhinovirus[at age 0-11, 12-59]
  51                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_hMPV[at age 0-11, 12-59]
  52                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_parainfluenza[at age 0-11, 12-59]
  53                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_streptococcus[at age 0-11, 12-59]
  54                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_hib[at age 0-11, 12-59]
  55                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_TB[at age 0-11, 12-59]
  56                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_staph[at age 0-11, 12-59]
  57                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_influenza[at age 0-11, 12-59]
  58                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          base\_incidence\_pneumonia\_by\_jirovecii[at age 0-11, 12-59]
  59  rr\_ri\_pneumonia\_Hhhandwashing                        relative rate of acquiring pneumonia for children with household handwashing with soap                                                                                                                                                                                                                                                                                                                                                                                                          independent risk factors assumed to have the same values for all pathogens for now
  60  rr\_ri\_pneumonia\_excl\_breastfeeding                  relative rate of acquiring pneumonia for children that are exclusively breastfeed
  61  rr\_ri\_pneumonia\_HIV                                  relative rate of pneumonia for HIV+ status
  62  rr\_ri\_pneumonia\_malnutrition                         relative rate of pneumonia for severe acute malnutrition
  63  rr\_ri\_pneumonia\_conti\_breast                        relative rate of pneumonia for not continued breastfeeding
  64  rr\_ri\_diarrhoea\_pneumococcal\_vaccination            relative rate of pneumonia for children that are vaccinated for pneumococcal                                                                                                                                                                                                                                                                                                                                                                                                                    only for strep. pneumoniae-cause pneumonia
  65  rr\_ri\_pneumonia\_hib\_vaccination                     relative rate of acquiring pneumonia for children that are vaccinated for Hib                                                                                                                                                                                                                                                                                                                                                                                                                   only for hib-cause pneumonia
  66  rr\_ri\_pneumonia\_influenza\_vaccination               relative rate of acquiring pneumonia for children that are vaccinated for influenza                                                                                                                                                                                                                                                                                                                                                                                                             only for influenza-cause pneumonia
  67  r\_progress\_to\_severe\_pneumonia                      rate of progression to severe penumonia in baseline group: age 2-11 months, no HIV, no SAM, bacterial-cause
  68  rr\_progress\_severe\_pneum\_age                        relative rate of progression to severe penumonia for ages 12-23, 24-59 months
  69  rr\_progress\_severe\_pneum\_HIV                        relative rate of progression to severe penumonia for HIV positive
  70  rr\_progress\_severe\_pneum\_SAM                        relative rate of progression to severe penumonia for severe acute malnutrition
  71  rr\_progress\_severe\_pneum\_viral                      relative rate of progression to severe penumonia for viral-cause pneumonia
  72  r\_progress\_to\_very\_sev\_pneumonia                   rate of progression to severe penumonia in baseline group: age 2-11 months, no HIV, no SAM, bacterial-cause
  73  rr\_progress\_very\_sev\_pneum\_age                     relative rate of progression to severe penumonia for ages 12-23, 24-59 months
  74  rr\_progress\_very\_sev\_pneum\_HIV                     relative rate of progression to severe penumonia for HIV positive
  75  rr\_progress\_very\_sev\_pneum\_SAM                     relative rate of progression to severe penumonia for severe acute malnutrition
  76  rr\_progress\_very\_sev\_pneum\_viral                   relative rate of progression to severe penumonia for viral-cause pneumonia
  77  r\_death\_pneumonia
  78                                                          relative rate of death from dysentery for ages 12-13, 24-59 months
  79                                                          relative rate of death from dysentery for HIV+ status
  80                                                          relative rate of death from dysentery for severe acute malnutrition
  81                                                          relative rate of death from dysentery for no treatment with antibiotics
  82  r\_death\_watery\_diarrhoea                             death rate from acute watery diarrhoea, baseline group: 0-11 months, HIV negative, no SAM, ORS given
  83  rr\_death\_watery\_diarrhoea\_age\_category             relative rate of death from acute watery diarrhoea for ages 12-13, 24-59 months
  84  rr\_death\_watery\_diarrhoea\_HIV                       relative rate of death from acute watery diarrhoea for HIV+ status
  85  rr\_death\_watery\_diarrhoea\_malnutrition              relative rate of death from acute watery diarrhoea for severe acute malnutrition
  86  r\_death\_persistent\_diarrhoea                         death rate from persistent diarrhoea, baseline group: 0-11 months, HIV negative, no SAM
  87  rr\_death\_persistent\_diarrhoea\_age\_category         relative rate of death from persistent diarrhoea for ages 12-13, 24-59 months
  88  rr\_death\_persistent\_diarrhoea\_HIV                   relative rate of death from persistent diarrhoea for HIV+ status
  89  rr\_death\_persistent\_diarrhoea\_malnutrition          relative rate of death from persistent diarrhoea for severe acute malnutrition
====  ======================================================  ==============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================  ==================================================================================  ============  ============

Parameter_values
----------------

====  ================================================  ===============================  ====================  ==================  ============  ===========================================  ====================  ====================
  ..  parameter\_name                                   value                            value2                value3              Unnamed: 4    Unnamed: 5                                   Unnamed: 6            Unnamed: 7
====  ================================================  ===============================  ====================  ==================  ============  ===========================================  ====================  ====================
   0  base\_incidence\_pneumonia\_by\_agecat            0.607                            0.3429                0.1204
   1  pn\_attributable\_fraction\_RSV                   0.397                            0.164                 0.165
   2  pn\_attributable\_fraction\_rhinovirus            0.029                            0.152                 0.159
   3  pn\_attributable\_fraction\_hmpv                  0.083                            0.063                 0.059
   4  pn\_attributable\_fraction\_parainfluenza         0.078                            0.067                 0.068
   5  pn\_attributable\_fraction\_streptococcus         0.047                            0.106                 0.092
   6  pn\_attributable\_fraction\_hib                   0.059                            0.067                 0.044
   7  pn\_attributable\_fraction\_TB                    0.067                            0.039                 0.059
   8  pn\_attributable\_fraction\_staph                 0.037                            0.009                 0.012
   9  pn\_attributable\_fraction\_influenza             0.016                            0.029                 0.032
  10  pn\_attributable\_fraction\_jirovecii             0.03                             0.004                 0.002
  11  pn\_attributable\_fraction\_other\_pathogens      0.098                            0.274                 0.282
  12  pn\_attributable\_fraction\_other\_cause          0.05899999999999994              0.025999999999999912  0.0259999999999998                Target values for 12 monthly incidence rate
  13  base\_inc\_rate\_pneumonia\_by\_RSV               [0.240979,0.0562356,0.019866]                                                            0.240979                                     0.0562356             0.019865999999999998
  14  base\_inc\_rate\_pneumonia\_by\_rhinovirus        [0.017603,0.0521208,0.0191436]                                                           0.017603                                     0.052120799999999995  0.0191436
  15  base\_inc\_rate\_pneumonia\_by\_hMPV              [0.050381,0.0216027,0.0071036]                                                           0.050381                                     0.0216027             0.007103599999999999
  16  base\_inc\_rate\_pneumonia\_by\_parainfluenza     [0.047346,0.0229743,0.0081872]                                                           0.047346                                     0.0229743             0.0081872
  17  base\_inc\_rate\_pneumonia\_by\_streptococcus     [0.028529,0.027915478,0.004884]                                                          0.028529                                     0.027915478           0.004884
  18  base\_inc\_rate\_pneumonia\_by\_hib               [0.035813,0.170477152,0.019758]                                                          0.035813                                     0.170477152           0.019758
  19  base\_inc\_rate\_pneumonia\_by\_TB                [0.040669,0.132603114,0.00555]                                                           0.040669000000000004                         0.132603114           0.00555
  20  base\_inc\_rate\_pneumonia\_by\_staphylococcus    [0.014689,0.066146727,0.000888]                                                          0.014689                                     0.066146727           0.000888
  21  base\_inc\_rate\_pneumonia\_by\_influenza         [0.000464,0.035974076,0.001332]                                                          0.00046400000000000006                       0.035974076           0.001332
  22  base\_inc\_rate\_pneumonia\_by\_jirovecii         [0.00249,0.022716889,0.001998]                                                           0.00249                                      0.022716889           0.001998
  23  base\_inc\_rate\_pneumonia\_by\_other\_pathogens  [0.059486,0.0939546,0.0339528]                                                           0.059486000000000004                         0.0939546             0.0339528
  24  rr\_ri\_pneumonia\_HHhandwashing                  0.7
  25  rr\_ri\_pneumonia\_indoor\_air\_pollution         0.8
  26  rr\_ri\_pneumonia\_excl\_breastfeeding            0.5
  27  rr\_ri\_pneumonia\_cont\_breast                   0.9
  28  rr\_ri\_pneumonia\_HIV                            1.3
  29  rr\_ri\_pneumonia\_SAM                            1.4
  30  rr\_ri\_pneumonia\_pneumococcal\_vaccine          0.5
  31  rr\_ri\_pneumonia\_hib\_vaccine                   0.5
  32  rr\_ri\_pneumonia\_influenza\_vaccine             0.5
  33  r\_progress\_to\_severe\_pneumonia                [0.2117, 0.17335, 0.11287]
  34  rr\_progress\_severe\_pneum\_age12to23mo          0.7
  35  rr\_progress\_severe\_pneum\_age24to59mo          0.2
  36  rr\_progress\_severe\_pneum\_HIV                  1.3
  37  rr\_progress\_severe\_pneum\_SAM                  1.4
  38  rr\_progress\_severe\_pneum\_viral                0.7
  39  r\_death\_pneumonia                               0.4
  40  rr\_death\_pneumonia\_agelt2mo                    1.4
  41  rr\_death\_pneumonia\_age12to23mo                 0.8
  42  rr\_death\_pneumonia\_age24to59mo                 0.3
  43  rr\_death\_pneumonia\_HIV                         1.4
  44  rr\_death\_pneumonia\_SAM                         1.4
====  ================================================  ===============================  ====================  ==================  ============  ===========================================  ====================  ====================

Pneumonia
---------

====  =====  =====  =========  ===  ===========================  ============  ============  ================================  ===========  ==================  =======================
  ..  Age    Sex    Smoking    …    Probability of Affliction    Unnamed: 5    Unnamed: 6    symptoms                          pneumonia    severe pneumonia    very severe pneumonia
====  =====  =====  =========  ===  ===========================  ============  ============  ================================  ===========  ==================  =======================
   0  15     Male   Yes             0.2                                                      fever
   1  16     Male   Yes             0.001                                                    cough and/or difficult breathing
   2  17     Male   Yes             0.1                                                      fast breathing
   3  18     Male   Yes             0                                                        chest indrawing
   4  15     Male   No              0                                                        stridor
   5  16     Male   No              0                                                        lethargic/unconscious
   6  17     Male   No              0                                                        not able to drink/ breastfeed
   7  18     Male   No              0                                                        convulsions
   8                                                                                         vomiting everything
====  =====  =====  =========  ===  ===========================  ============  ============  ================================  ===========  ==================  =======================

Severe pneumonia
----------------

====  =====  =====  =========  ===  ===========================
  ..    Age  Sex    Smoking    …      Probability of Affliction
====  =====  =====  =========  ===  ===========================
   0     15  Male   Yes                                   0.2
   1     16  Male   Yes                                   0.001
   2     17  Male   Yes                                   0.1
   3     18  Male   Yes                                   0
   4     15  Male   No                                    0
   5     16  Male   No                                    0
   6     17  Male   No                                    0
   7     18  Male   No                                    0
====  =====  =====  =========  ===  ===========================

Very severe pneumonia
---------------------

====  ============================================  ==================================================================================================================  ===================  ===========================================================================================
  ..  Parameters                                    Description                                                                                                         Value                Reference and Notes
====  ============================================  ==================================================================================================================  ===================  ===========================================================================================
   0  base\_incidence\_pneumonia\_by\_agecat        incidence of pneumonia in each age group: [0-11, 12-23, 24-59 months]                                               0.666, 0.367, 0.127  Yearly incidence rate in cases per child, calculated from the McCollum ED et al, 2017 data.
   1  rr\_ri\_pneumonia\_HHhandwashing                                                                                                                                  0.7
   2  rr\_ri\_pneumonia\_indoor\_air\_pollution                                                                                                                         0.8
   3  rr\_ri\_pneumonia\_excl\_breastfeeding                                                                                                                            0.5
   4  rr\_ri\_pneumonia\_cont\_breast                                                                                                                                   0.9
   5  rr\_ri\_pneumonia\_HIV                                                                                                                                            1.1
   6  rr\_ri\_pneumonia\_malnutrition                                                                                                                                   1.4
   7  rr\_ri\_pneumonia\_pneumococcal\_vaccine                                                                                                                          0.5
   8  rr\_ri\_pneumonia\_hib\_vaccine                                                                                                                                   0.5
   9  rr\_ri\_pneumonia\_influenza\_vaccine                                                                                                                             0.5
  10  r\_progress\_to\_severe\_penum                baseline rate of progression of non-severe to severe or very severe pneumonia for ages [0-11, 12-23, 24-59 months]  0.67, 0.62, 0.47
  11  r\_progress\_to\_very\_sev\_penum             baseline rate of progression of severe to very severe pneumonia for ages [0-11, 12-23, 24-59 months]                0.27, 0.26, 0.195
  12  pn\_attributable\_fraction\_RSV               attributable fraction of RSV in age groups: [0-11, 12-23, 24-59 months]                                             0.397, 0.164, 0.165
  13  pn\_attributable\_fraction\_rhinovirus        attributable fraction of rhinovirus in age groups: [0-11, 12-23, 24-59 months]                                      0.029, 0.152, 0.159
  14  pn\_attributable\_fraction\_hmpv              attributable fraction of hMPV in age groups: [0-11, 12-23, 24-59 months]                                            0.083, 0.063, 0.059
  15  pn\_attributable\_fraction\_parainfluenza     attributable fraction of parainfluenza in age groups: [0-11, 12-23, 24-59 months]                                   0.078, 0.067, 0.068
  16  pn\_attributable\_fraction\_streptococcus     attributable fraction of streptococcus in age groups: [0-11, 12-23, 24-59 months]                                   0.047, 0.106, 0.092
  17  pn\_attributable\_fraction\_hib               attributable fraction of Hib in age groups: [0-11, 12-23, 24-59 months]                                             0.059, 0.067, 0.044
  18  pn\_attributable\_fraction\_TB                attributable fraction of TB in age groups: [0-11, 12-23, 24-59 months]                                              0.067, 0.039, 0.059  need to call from TB module those infected.
  19  pn\_attributable\_fraction\_staph             attributable fraction of staphylococcus in age groups: [0-11, 12-23, 24-59 months]                                  0.037, 0.009, 0.012
  20  pn\_attributable\_fraction\_influenza         attributable fraction of influenza in age groups: [0-11, 12-23, 24-59 months]                                       0.016, 0.029, 0.032
  21  pn\_attributable\_fraction\_jirovecii         attributable fraction of P. jirovecii in age groups: [0-11, 12-23, 24-59 months]                                    0.03, 0.004, 0.002
  22  pn\_attributable\_fraction\_other\_pathogens  attributable fraction of other pathogens in age groups: [0-11, 12-23, 24-59 months]                                 0.098, 0.274, 0.282
  23  pn\_attributable\_fraction\_other\_cause      attributable fraction of other causes in age groups: [0-11, 12-23, 24-59 months]                                    0.059, 0.026, 0.026
  24  rr\_prog\_very\_sev\_pneum\_RSV               relative progression to very severe pneumonia for RSV                                                               0.7159
  25  rr\_prog\_very\_sev\_pneum\_rhinovirus        relative progression to very severe pneumonia for rhinovirus                                                        0.9506
  26  rr\_prog\_very\_sev\_pneum\_hmpv              relative progression to very severe pneumonia for Hmpv                                                              0.9512
  27  rr\_prog\_very\_sev\_pneum\_parainfluenza     relative progression to very severe pneumonia for parainfluenza                                                     0.5556
  28  rr\_prog\_very\_sev\_pneum\_streptococcus     relative progression to very severe pneumonia for streptococcus                                                     2.1087
  29  rr\_prog\_very\_sev\_pneum\_hib               relative progression to very severe pneumonia for hib                                                               1.6122
  30  rr\_prog\_very\_sev\_pneum\_TB                relative progression to very severe pneumonia for TB                                                                1.1667
  31  rr\_prog\_very\_sev\_pneum\_staph             relative progression to very severe pneumonia for staph                                                             5.2727
  32  rr\_prog\_very\_sev\_pneum\_influenza         relative progression to very severe pneumonia for influenza                                                         1.4
  33  rr\_prog\_very\_sev\_pneum\_jirovecii         relative progression to very severe pneumonia for P. jirovecii                                                      1.9167
  34
  35
  36
  37
  38
  39
  40
  41
  42
  43
  44
  45
  46  rr\_progress\_severe\_pneum\_HIV              1.3
  47  rr\_progress\_severe\_pneum\_SAM              1.4
  48
  49
  50
  51
  52  rr\_progress\_very\_sev\_pneum\_HIV           1.3
  53  rr\_progress\_very\_sev\_pneum\_SAM           1.4
  54
  55  r\_death\_pneumonia                           0.4
  56  rr\_death\_pneumonia\_HIV                     0.45
====  ============================================  ==================================================================================================================  ===================  ===========================================================================================

_Parameter1
-----------

====  =====  =====  =========  ===  ============  ============================
  ..    Age  Sex    Smoking    …    Assignment      Probability of Assignement
====  =====  =====  =========  ===  ============  ============================
   0     15  Male   Yes                                                  0.2
   1     16  Male   Yes                                                  0.001
   2     17  Male   Yes                                                  0.1
   3     18  Male   Yes                                                  0
   4     15  Male   No                                                   0
   5     16  Male   No                                                   0
   6     17  Male   No                                                   0
   7     18  Male   No                                                   0
====  =====  =====  =========  ===  ============  ============================

References
----------



