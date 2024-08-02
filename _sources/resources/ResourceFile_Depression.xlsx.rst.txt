Depression (.xlsx)
==================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Depression.xlsx>`

.. contents::

parameter_values
----------------

====  ========================================================  ==========================  ============  =======================================================================================================================================================================================================================================================================================================================================  ============  ============  ============  ============
  ..  parameter\_name                                           value                       Unnamed: 2    Justification                                                                                                                                                                                                                                                                                                                            Unnamed: 4    Unnamed: 5    Unnamed: 6    Unnamed: 7
====  ========================================================  ==========================  ============  =======================================================================================================================================================================================================================================================================================================================================  ============  ============  ============  ============
   0  init\_pr\_depr\_m\_age1519\_no\_cc\_wealth123             0.06                                      None
   1  init\_rp\_depr\_f\_not\_rec\_preg                         1.5                                       None
   2  init\_rp\_depr\_f\_rec\_preg                              3                                         None
   3  init\_rp\_depr\_age2059                                   1                                         None
   4  init\_rp\_depr\_agege60                                   3                                         None
   5  init\_rp\_depr\_cc                                        1                                         None
   6  init\_rp\_depr\_wealth45                                  1                                         None
   7  init\_rp\_ever\_depr\_per\_year\_older\_m                 0.004                                     None
   8  init\_rp\_ever\_depr\_per\_year\_older\_f                 0.006                                     None
   9  init\_pr\_antidepr\_curr\_depr                            0.11                                      None
  10  init\_rp\_antidepr\_ever\_depr\_not\_curr                 0.5                                       None
  11  init\_pr\_ever\_diagnosed\_depression                     0.2                                       None
  12  init\_pr\_ever\_talking\_therapy\_if\_diagnosed           1                                         We assume that talking therapy happens as part of diagnosis
  13  init\_pr\_ever\_self\_harmed\_if\_ever\_depr              0.004                                     consistent with rate of incident self harm
  14  base\_3m\_prob\_depr                                      0.0021                                    Rate is derived indirectly based on prevalence of depression that model produces as an output (Abas et al, 1997; Marwick et al, 2010; Udedi et al, 2014; This is much higher than incidence in Todd et al 1999 (approx 0.16 in 2 months) but that is inconsistent with prevalence unless extremely short duration. )
  15  rr\_depr\_wealth45                                        3                                         Abas et al, 1997; Todd et al 1999; Chibanda et al, 2012;
  16  rr\_depr\_cc                                              1.25                                      HIV:  Cohen et al Rwanda 2009; Brandt 2009; hypertension: no association in SA  Grimsrud et al 2009.  Likely the association is with chronic symptoms rather than diagnosis of a chronic condition.
  17  rr\_depr\_pregnancy                                       3                                         could not identify any estimates for African countries; although clear that pregnancy prevalence is relatively high in pregnancy and post-partum (it is clear the relative rate is > 1).  Suggest use value of 3 until we have more data.
  18  rr\_depr\_female                                          1.5                                       Kohler et al 2017
  19  rr\_depr\_prev\_epis                                      50                                        No data identified from Malawi or other close countries, but likely to be a large person-specific effect.  Consider data on proportion of people with depression ever.
  20  rr\_depr\_on\_antidepr                                    30
  21  rr\_depr\_age1519                                         1                                         Kim et al 2015;
  22  rr\_depr\_agege60                                         3                                         Kohler et al 2017;
  23  rr\_depr\_hiv                                             1.99                                      Ciesla & Roberts 2001;
  24  depr\_resolution\_rates                                   [0.2, 0.3, 0.5, 0.7, 0.95]                Abas et al, 1997; Dow et al 2014;
  25  rr\_resol\_depr\_cc                                       0.5
  26  rr\_resol\_depr\_on\_antidepr                             1.5                                       Hengartner 2017; Cipriani et al 2018; Furukawa et al 2016;Faria et 2017; Kirsch 2014;
  27  rr\_resol\_depr\_current\_talk\_ther                      1.1
  28  prob\_3m\_stop\_antidepr                                  0.7                                       None
  29  prob\_3m\_default\_antidepr                               0.2                                       None
  30  prob\_3m\_suicide\_depr\_m                                0.0005                                    Chasimpha et al 2015   to be infered based on overall suicide rate and prevalence of depression (assume depression a pre-requisite for suicide) Suicide rate in adults in Karonga study 26.1 per 100,000 person years in adult men (age ge 15) and 8.0 per 100,000 person years in adult women.  Chasimpha et al BMC Public Health 2015
  31  rr\_suicide\_depr\_f                                      0.333                                     Chasimpha et al 2015
  32  prob\_3m\_selfharm\_depr                                  0.0005                                    None
  33  sensitivity\_of\_assessment\_of\_depression               0.75                                      None
  34  pr\_assessed\_for\_depression\_in\_generic\_appt\_level1  0.01                                      No available reference found, but co-calibrated with pr\_assessed\_for\_depression\_for\_perinatal\_female for a better Model/Data ratio on mental health service usage.
  35  anti\_depressant\_medication\_item\_code                  267                                       First line treatment (Amitriptyline 25mg\_100\_CMST)
  36  pr\_assessed\_for\_depression\_for\_perinatal\_female     0.01                                      No available reference found, but co-calibrated with pr\_assessed\_for\_depression\_in\_generic\_appt\_level1 for a better Model/Data ratio on mental health service usage.
====  ========================================================  ==========================  ============  =======================================================================================================================================================================================================================================================================================================================================  ============  ============  ============  ============

