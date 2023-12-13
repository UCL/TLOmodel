Schisto (.xlsx)
===============

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Schisto.xlsx>`

.. contents::

Parameters
----------

====  =============================================  =======  ===================  ==================
  ..  Parameter                                        Value  Reference            Area
====  =============================================  =======  ===================  ==================
   0  delay\_till\_hsi\_a\_repeated                   5       assumption           HSI
   1  delay\_till\_hsi\_b\_repeated                  20       assumption           HSI
   2  prob\_sent\_to\_lab\_test\_children             0.9     assumption           HSI
   3  prob\_sent\_to\_lab\_test\_adults               0.6     assumption           HSI
   4  PZQ\_efficacy                                   1       assumption           HSI
   5  high\_intensity\_threshold\_haematobium        20       WHO                  transmission model
   6  high\_intensity\_threshold\_PSAC\_haematobium   5       assumption           transmission model
   7  low\_intensity\_threshold\_haematobium          2       assumption           transmission model
   8  worms\_fecundity\_haematobium                   0.005   Truscott et al       transmission model
   9  worm\_lifespan\_haematobium                     6       Anderson et al 2016  transmission model
  10  beta\_PSAC\_haematobium                         0.3     Truscott et al       transmission model
  11  beta\_SAC\_haematobium                          1       Truscott et al       transmission model
  12  beta\_Adults\_haematobium                       0.05    Truscott et al       transmission model
  13  high\_intensity\_threshold\_mansoni            46       WHO                  transmission model
  14  high\_intensity\_threshold\_PSAC\_mansoni      20       assumption           transmission model
  15  low\_intensity\_threshold\_mansoni              2       assumption           transmission model
  16  worms\_fecundity\_mansoni                       0.0006  Chan et al           transmission model
  17  worm\_lifespan\_mansoni                         4       Anderson et al 2017  transmission model
  18  beta\_PSAC\_mansoni                             0.3     Truscott et al       transmission model
  19  beta\_SAC\_mansoni                              1       Truscott et al       transmission model
  20  beta\_Adults\_mansoni                           0.05    Truscott et al       transmission model
====  =============================================  =======  ===================  ==================

Symptoms
--------

====  ==================  ======================  ============  =========================  =================
  ..  Symptom             HSB\_mapped\_symptom      Prevalence  Reference                  Infection\_type
====  ==================  ======================  ============  =========================  =================
   0  anemia              other                         0.9     assumption                 both
   1  fever               fever                         0.3     assumption                 both
   2  haematuria          other                         0.625   van der Werf et al., 2003  haematobium
   3  hydronephrosis      stomach\_ache                 0.083   van der Werf et al., 2003  haematobium
   4  dysuria             stomach\_ache                 0.2857  van der Werf et al., 2003  haematobium
   5  bladder\_pathology  stomach\_ache                 0.7857  van der Werf et al., 2003  haematobium
   6  ascites             stomach\_ache                 0.0054  van der Werf et al., 2003  mansoni
   7  diarrhoea           diarrhoea                     0.0144  van der Werf et al., 2003  mansoni
   8  vomiting            vomit                         0.0172  van der Werf et al., 2003  mansoni
   9  hepatomegaly        stomach\_ache                 0.1574  van der Werf et al., 2003  mansoni
====  ==================  ======================  ============  =========================  =================

DALYs
-----

====  ====================  ========================================  =======================================  ================================================================================================================================  ===================  =======  ========
  ..    TLO\_Sequela\_Code  Sequela                                   Health state name                        Health state lay description                                                                                                        disability weight    lower    uppper
====  ====================  ========================================  =======================================  ================================================================================================================================  ===================  =======  ========
   0                   254  Mild diarrhea due to schistosomiasis      Diarrhea, mild                           has diarrhea three or more times a day with occasional discomfort in the belly.                                                                 0.074    0.049     0.104
   1                   255  Severe anemia due to schistosomiasis      Anemia, severe                           feels very weak, tired and short of breath, and has problems with activities that require physical effort or deep concentration.                0.149    0.101     0.209
   2                   256  Hematemesis due to schistosomiasis        Gastric bleeding                         vomits blood and feels nauseous.                                                                                                                0.325    0.209     0.462
   3                   257  Hepatomegaly due to schistosomiasis       Abdominopelvic problem, mild             has some pain in the belly that causes nausea but does not interfere with daily activities.                                                     0.011    0.005     0.021
   4                   258  Moderate anemia due to schistosomiasis    Anemia, moderate                         feels moderate fatigue, weakness, and shortness of breath after exercise, making daily activities more difficult.                               0.052    0.034     0.076
   5                   259  Mild anemia due to schistosomiasis        Anemia, mild                             feels slightly tired and weak at times, but this does not interfere with normal daily activities.                                               0.004    0.001     0.008
   6                   260  Hydronephrosis due to schistosomiasis     Abdominopelvic problem, mild             has some pain in the belly that causes nausea but does not interfere with daily activities.                                                     0.011    0.005     0.021
   7                   261  Ascites due to schistosomiasis            Abdominopelvic problem, moderate         has pain in the belly and feels nauseous. The person has difficulties with daily activities.                                                    0.114    0.078     0.159
   8                   262  Mild schistosomiasis                      Infectious disease, acute episode, mild  has a low fever and mild discomfort , but no difficulty with daily activities.                                                                  0.006    0.002     0.012
   9                   263  Dysuria due to schistosomiasis            Abdominopelvic problem, mild             has some pain in the belly that causes nausea but does not interfere with daily activities.                                                     0.011    0.005     0.021
  10                   264  Bladder pathology due to schistosomiasis  Abdominopelvic problem, mild             has some pain in the belly that causes nausea but does not interfere with daily activities.                                                     0.011    0.005     0.021
====  ====================  ========================================  =======================================  ================================================================================================================================  ===================  =======  ========

District_Params_haematobium
---------------------------

====  =============  ========  ===========  ===========  ==============  ===========  ============  ====================================================
  ..  District       Region      Reservoir    R0\_value    alpha\_value  Reference      Prevalence  Reference\_prevalence
====  =============  ========  ===========  ===========  ==============  ===========  ============  ====================================================
   0  Balaka         Southern      0            1.12425       0.061841   fitted          0.154602   2011 SCI / MoH, ESPEN
   1  Blantyre       Southern      1.87216      1.12956       0.107616   fitted          0.26904    rural, 2011 SCI / MoH. ESPEN
   2  Blantyre City  Southern      0            1.1253        0.04664    fitted          0.1166     2011 SCI / MoH
   3  Chikwawa       Southern      0            1.1248        0.0805     fitted          0.20125    2008 WFP / MoH, ESPEN
   4  Chiradzulu     Southern      2.81222      1.14141       0.1377     fitted          0.34425    2010 MoH, ESPEN
   5  Chitipa        Northern      0            1.12577       0.08905    fitted          0.222625   2010 MoH, ESPEN
   6  Dedza          Central       0            1.12423       0.0698     fitted          0.1745     2010 MoH, ESPEN
   7  Dowa           Central       0            0             0          fitted          0          ESPEN
   8  Karonga        Northern      0            1.12421       0.0643     fitted          0.16075    2010 COM
   9  Kasungu        Central       0            1.12471       0.0793714  fitted          0.198429   2008 WFP/MoH, 2010 MoH, ESPEN
  10  Likoma         Northern      0            1.13144       0.114      fitted          0.285      ESPEN
  11  Lilongwe       Central       0            1.12704       0.0967167  fitted          0.241792   2008 WFP / MoH, Jemu SK et al, 2011 SCI/MoH, , ESPEN
  12  Lilongwe City  Central       0            1.12437       0.05852    fitted          0.1463     2011 SCI / MoH
  13  Machinga       Southern      0            1.12429       0.0718     fitted          0.1795     2010 MoH, ESPEN
  14  Mangochi       Southern      0            1.12421       0.0643333  fitted          0.160833   ESPEN
  15  Mchinji        Central       0            1.12448       0.076      fitted          0.19       2010 MoH, ESPEN
  16  Mulanje        Southern      3.05         1.14496       0.144      fitted          0.36       2010 MoH, ESPEN
  17  Mwanza         Southern      0            1.12506       0.0832457  fitted          0.208114   2011 SCI / MoH, ESPEN
  18  Mzimba         Northern      0            1.12536       0.046129   fitted          0.115323   2010 MoH, ESPEN
  19  Mzuzu City     Northern      0            1.13018       0.02004    fitted          0.0501     2011 SCI / MoH
  20  Neno           Southern      0            1.12631       0.03896    fitted          0.0974     2011 SCI / MoH, ESPEN
  21  Nkhata Bay     Northern      0            1.13233       0.01225    fitted          0.030625   ESPEN
  22  Nkhotakota     Central       2.19995      1.13321       0.119188   fitted          0.297971   2010 MoH, ESPEN
  23  Nsanje         Southern      2.39017      1.13559       0.125333   fitted          0.313333   2008 WFP / MoH, ESPEN
  24  Ntcheu         Central       0            1.12421       0.0684143  fitted          0.171036   2008 WFP/MoH, 2010 MoH, ESPEN
  25  Ntchisi        Central       0            1.12433       0.073      fitted          0.1825     ESPEN
  26  Phalombe       Southern      5.31723      1.1841        0.188      fitted          0.47       ESPEN
  27  Rumphi         Northern      0            0             0          fitted          0          ESPEN
  28  Salima         Central       0            1.12873       0.0261176  fitted          0.0652941  ESPEN
  29  Thyolo         Southern      0            1.12421       0.06865    fitted          0.171625   2010 MoH, ESPEN
  30  Zomba          Southern      0            1.1262        0.0397333  fitted          0.0993333  ESPEN
  31  Zomba City     Southern      0            1.1262        0.0397333  fitted          0.0993333  ESPEN copy from Zomba
====  =============  ========  ===========  ===========  ==============  ===========  ============  ====================================================

District_Params_mansoni
-----------------------

====  =============  ========  ===========  ===========  ==============  ===========  ============  ===============================
  ..  District       Region      Reservoir    R0\_value    alpha\_value  Reference      Prevalence  Reference\_prevalence
====  =============  ========  ===========  ===========  ==============  ===========  ============  ===============================
   0  Balaka         Southern    0.0103949      3.28891       0.0528333  fitted         0.00944444  ESPEN
   1  Blantyre       Southern    0.0796066      1.46175       0.0652093  fitted         0.0506977   ESPEN
   2  Blantyre City  Southern    0.0796066      1.46175       0.0652093  fitted         0.0506977   copy from Blantyre
   3  Chikwawa       Southern    0.0243277      2.18968       0.056      fitted         0.02        ESPEN
   4  Chiradzulu     Southern    0.158777       1.26817       0.07445    fitted         0.0815      ESPEN
   5  Chitipa        Northern    0              0             0.05       fitted         0           ESPEN
   6  Dedza          Central     0              0             0.05       fitted         0           ESPEN
   7  Dowa           Central     0              0             0.05       fitted         0           ESPEN
   8  Karonga        Northern    0.0891927      1.42196       0.0665     fitted         0.055       average of all mansoni in Espen
   9  Kasungu        Central     0              0             0.05       fitted         0           ESPEN
  10  Likoma         Northern    1.12359        1.06177       0.125      fitted         0.25        ESPEN
  11  Lilongwe       Central     0.187713       1.23539       0.0772459  fitted         0.0908197   ESPEN
  12  Lilongwe City  Central     0.187713       1.23539       0.0772459  fitted         0.0908197   copy from Lilongwe
  13  Machinga       Southern    0              0             0.05       fitted         0           ESPEN
  14  Mangochi       Southern    0.114042       1.34757       0.0695882  fitted         0.0652941   ESPEN
  15  Mchinji        Central     0              0             0.05       fitted         0           ESPEN
  16  Mulanje        Southern    0              0             0.05       fitted         0           ESPEN
  17  Mwanza         Southern    0.0411611      1.78166       0.0592308  fitted         0.0307692   ESPEN
  18  Mzimba         Northern    0.0236342      2.21729       0.0558548  fitted         0.0195161   ESPEN
  19  Mzuzu City     Northern    0.0891927      1.42196       0.0665     fitted         0.055       average of all mansoni in Espen
  20  Neno           Southern    0              0             0.05       fitted         0           ESPEN
  21  Nkhata Bay     Northern    0.0377989      1.83686       0.058625   fitted         0.02875     ESPEN
  22  Nkhotakota     Central     0.172451       1.25145       0.0758     fitted         0.086       ESPEN
  23  Nsanje         Southern    0              0             0.05       fitted         0           ESPEN
  24  Ntcheu         Central     0.0194311      2.42068       0.05495    fitted         0.0165      ESPEN
  25  Ntchisi        Central     0.203152       1.22138       0.07865    fitted         0.0955      ESPEN
  26  Phalombe       Southern    0              0             0.05       fitted         0           ESPEN
  27  Rumphi         Northern    0              0             0.05       fitted         0           ESPEN
  28  Salima         Central     0.190781       1.23244       0.0775294  fitted         0.0917647   ESPEN
  29  Thyolo         Southern    0              0             0.05       fitted         0           ESPEN
  30  Zomba          Southern    0.172451       1.25145       0.0758     fitted         0.086       ESPEN
  31  Zomba City     Southern    0.172451       1.25145       0.0758     fitted         0.086       copy from Zomba
====  =============  ========  ===========  ===========  ==============  ===========  ============  ===============================

MDA_historical_Coverage
-----------------------

====  =============  ========  ======  ===============  ==============  =================  ========================================================================
  ..  District       Region      Year    Coverage PSAC    Coverage SAC    Coverage Adults  Reference
====  =============  ========  ======  ===============  ==============  =================  ========================================================================
   0  Blantyre       Southern    2015                0           0.78               0      data
   1  Blantyre       Southern    2016                0           0.85               0.44   interpolated, mean of 2015 and 2017
   2  Blantyre       Southern    2017                0           0.92               0.88   data
   3  Blantyre       Southern    2018                0           0.87               0.88   data
   4  Chiradzulu     Southern    2015                0           0.73               0.02   data
   5  Chiradzulu     Southern    2016                0           0.8                0.155  interpolated, mean of 2015 and 2017
   6  Chiradzulu     Southern    2017                0           0.87               0.29   value over 100% in the data for SAC so instead we use 2018 value
   7  Chiradzulu     Southern    2018                0           0.87               0.9    data
   8  Mulanje        Southern    2015                0           0.48               0.96   data
   9  Mulanje        Southern    2016                0           0.605              0.96   interpolated, mean of 2015 and 2017
  10  Mulanje        Southern    2017                0           0.73               0.96   values over 100% in the data so instead we used values from 2018
  11  Mulanje        Southern    2018                0           0.73               0.96   value over 100% in the data for Adults so instead we use value for 2015
  12  Nkhotakota     Central     2015                0           0.5                0.01   value over 100% in the data for SAC so instead we used 0.5
  13  Nkhotakota     Central     2016                0           0.675              0.22   interpolated, mean of 2015 and 2017
  14  Nkhotakota     Central     2017                0           0.85               0.43   value over 100% in the data for SAC so instead we used 2018 value
  15  Nkhotakota     Central     2018                0           0.85               0.89   data
  16  Nsanje         Southern    2015                0           0.75               0      data
  17  Nsanje         Southern    2016                0           0.795              0.41   interpolated, mean of 2015 and 2017
  18  Nsanje         Southern    2017                0           0.84               0.82   data
  19  Nsanje         Southern    2018                0           0.9                0.88   data
  20  Phalombe       Southern    2015                0           0.92               0.04   data
  21  Phalombe       Southern    2016                0           0.88               0.445  interpolated, mean of 2015 and 2017
  22  Phalombe       Southern    2017                0           0.84               0.85   data
  23  Phalombe       Southern    2018                0           0.86               0.92   data
  24  Balaka         Southern    2015                0           0.43               0      data
  25  Balaka         Southern    2016                0           0.65               0.44   interpolated, mean of 2015 and 2017
  26  Balaka         Southern    2017                0           0.87               0.88   data
  27  Balaka         Southern    2018                0           0.83               0.89   data
  28  Blantyre City  Southern    2015                0           0.78               0      data
  29  Blantyre City  Southern    2016                0           0.85               0.44   interpolated, mean of 2015 and 2017
  30  Blantyre City  Southern    2017                0           0.92               0.88   data
  31  Blantyre City  Southern    2018                0           0.87               0.88   data
  32  Chikwawa       Southern    2015                0           0.65               0      data
  33  Chikwawa       Southern    2016                0           0.765              0.49   interpolated, mean of 2015 and 2017
  34  Chikwawa       Southern    2017                0           0.88               0.98   data
  35  Chikwawa       Southern    2018                0           0.74               0.52   data
  36  Chitipa        Northern    2015                0           0.62               0      data
  37  Chitipa        Northern    2016                0           0.74               0.345  interpolated, mean of 2015 and 2017
  38  Chitipa        Northern    2017                0           0.86               0.69   errors in the data so instead we use2018 data
  39  Chitipa        Northern    2018                0           0.86               0.69   data
  40  Dedza          Central     2015                0           0.72               0      data
  41  Dedza          Central     2016                0           0.81               0.445  interpolated, mean of 2015 and 2017
  42  Dedza          Central     2017                0           0.9                0.89   data
  43  Dedza          Central     2018                0           0.95               0.95   data
  44  Dowa           Central     2015                0           0.85               0.02   data
  45  Dowa           Central     2016                0           0.85               0.435  interpolated, mean of 2015 and 2017
  46  Dowa           Central     2017                0           0.85               0.85   data
  47  Dowa           Central     2018                0           0.8                0.77   data
  48  Karonga        Northern    2015                0           0.97               0      data
  49  Karonga        Northern    2016                0           0.91               0.435  interpolated, mean of 2015 and 2017
  50  Karonga        Northern    2017                0           0.85               0.87   data
  51  Karonga        Northern    2018                0           0.89               0.84   data
  52  Kasungu        Central     2015                0           0.89               0      data
  53  Kasungu        Central     2016                0           0.88               0.35   interpolated, mean of 2015 and 2017
  54  Kasungu        Central     2017                0           0.87               0.7    data
  55  Kasungu        Central     2018                0           0.82               0.86   data
  56  Likoma         Northern    2015                0           0.52               0.26   data
  57  Likoma         Northern    2016                0           0.685              0.615  interpolated, mean of 2015 and 2017
  58  Likoma         Northern    2017                0           0.85               0.97   no data so we used Lilongwe data for 2017 and 2018
  59  Likoma         Northern    2018                0           0.83               0.9    no data, random
  60  Lilongwe       Central     2015                0           0.88               0.04   data
  61  Lilongwe       Central     2016                0           0.865              0.505  interpolated, mean of 2015 and 2017
  62  Lilongwe       Central     2017                0           0.85               0.97   data
  63  Lilongwe       Central     2018                0           0.83               0.9    data
  64  Lilongwe City  Central     2015                0           0.88               0.04   data
  65  Lilongwe City  Central     2016                0           0.865              0.505  interpolated, mean of 2015 and 2017
  66  Lilongwe City  Central     2017                0           0.85               0.97   data
  67  Lilongwe City  Central     2018                0           0.83               0.9    data
  68  Machinga       Southern    2015                0           0.45               0      data
  69  Machinga       Southern    2016                0           0.68               0.46   interpolated, mean of 2015 and 2017
  70  Machinga       Southern    2017                0           0.91               0.92   data
  71  Machinga       Southern    2018                0           0.93               0.97   data
  72  Mangochi       Southern    2015                0           0.82               0.01   data
  73  Mangochi       Southern    2016                0           0.85               0.49   interpolated, mean of 2015 and 2017
  74  Mangochi       Southern    2017                0           0.88               0.97   data
  75  Mangochi       Southern    2018                0           0.73               0.75   data
  76  Mchinji        Central     2015                0           0.77               0      data
  77  Mchinji        Central     2016                0           0.815              0.435  interpolated, mean of 2015 and 2017
  78  Mchinji        Central     2017                0           0.86               0.87   data
  79  Mchinji        Central     2018                0           0.83               0.82   data
  80  Mwanza         Southern    2015                0           0.5                0      errors in the data for SAC so instead we used 0.5
  81  Mwanza         Southern    2016                0           0.675              0.415  interpolated, mean of 2015 and 2017
  82  Mwanza         Southern    2017                0           0.85               0.83   data
  83  Mwanza         Southern    2018                0           0.75               0.72   data
  84  Mzimba         Northern    2015                0           0.41               0.03   data for Mzimba South
  85  Mzimba         Northern    2016                0           0.7                0.48   interpolated, mean of 2015 and 2017
  86  Mzimba         Northern    2017                0           0.99               0.93   value over 100% in the data for Adults so instead we used value for 2018
  87  Mzimba         Northern    2018                0           0.78               0.93   data for Mzimba South
  88  Mzuzu City     Northern    2015                0           0.83               0      no data, so copied from another Northern with data, Rumphi
  89  Mzuzu City     Northern    2016                0           0.855              0.435  no data, so copied from another Northern with data, Rumphi
  90  Mzuzu City     Northern    2017                0           0.88               0.87   no data, so copied from another Northern with data, Rumphi
  91  Mzuzu City     Northern    2018                0           0.92               0.89   no data, so copied from another Northern with data, Rumphi
  92  Neno           Southern    2015                0           0.8                0      data
  93  Neno           Southern    2016                0           0.82               0.395  interpolated, mean of 2015 and 2017
  94  Neno           Southern    2017                0           0.84               0.79   data
  95  Neno           Southern    2018                0           0.87               0.87   data
  96  Nkhata Bay     Northern    2015                0           0.5                0.76   value over 100% in the data for SAC so instead we used 0.5
  97  Nkhata Bay     Northern    2016                0           0.66               0.765  interpolated, mean of 2015 and 2017
  98  Nkhata Bay     Northern    2017                0           0.82               0.77   data
  99  Nkhata Bay     Northern    2018                0           0.81               0.8    data
 100  Ntcheu         Central     2015                0           0.91               0      data
 101  Ntcheu         Central     2016                0           0.865              0.415  interpolated, mean of 2015 and 2017
 102  Ntcheu         Central     2017                0           0.82               0.83   data
 103  Ntcheu         Central     2018                0           0.84               0.87   data
 104  Ntchisi        Central     2015                0           0.5                0      value over 100% in the data for SAC so instead we used 0.5
 105  Ntchisi        Central     2016                0           0.66               0.45   interpolated, mean of 2015 and 2017
 106  Ntchisi        Central     2017                0           0.82               0.9    data
 107  Ntchisi        Central     2018                0           0.87               0.83   data
 108  Rumphi         Northern    2015                0           0.83               0      data
 109  Rumphi         Northern    2016                0           0.855              0.435  interpolated, mean of 2015 and 2017
 110  Rumphi         Northern    2017                0           0.88               0.87   data
 111  Rumphi         Northern    2018                0           0.92               0.89   data
 112  Salima         Central     2015                0           0.5                0.33   value over 100% in the data for SAC so instead we used 0.5
 113  Salima         Central     2016                0           0.675              0.6    interpolated, mean of 2015 and 2017
 114  Salima         Central     2017                0           0.85               0.87   data
 115  Salima         Central     2018                0           0.85               0.78   value over 100% in the data for SAC so instead we used 2017 value
 116  Thyolo         Southern    2015                0           0.63               0.04   data
 117  Thyolo         Southern    2016                0           0.745              0.395  interpolated, mean of 2015 and 2017
 118  Thyolo         Southern    2017                0           0.86               0.75   data
 119  Thyolo         Southern    2018                0           0.88               0.85   data
 120  Zomba          Southern    2015                0           0.71               0      data
 121  Zomba          Southern    2016                0           0.81               0.39   interpolated, mean of 2015 and 2017
 122  Zomba          Southern    2017                0           0.91               0.78   data
 123  Zomba          Southern    2018                0           0.91               0.87   data
 124  Zomba City     Southern    2015                0           0.71               0      data
 125  Zomba City     Southern    2016                0           0.81               0.39   interpolated, mean of 2015 and 2017
 126  Zomba City     Southern    2017                0           0.91               0.78   data
 127  Zomba City     Southern    2018                0           0.91               0.87   data
====  =============  ========  ======  ===============  ==============  =================  ========================================================================

MDA_prognosed_Coverage
----------------------

====  =============  ========  ===========  ===============  ==============  =================
  ..  District       Region      Frequency    Coverage PSAC    Coverage SAC    Coverage Adults
====  =============  ========  ===========  ===============  ==============  =================
   0  Blantyre       Southern           12              0.5             0.8                0.5
   1  Chiradzulu     Southern           12              0.5             0.8                0.5
   2  Mulanje        Southern           12              0.5             0.8                0.5
   3  Nkhotakota     Central            12              0.5             0.8                0.5
   4  Nsanje         Southern           12              0.5             0.8                0.5
   5  Phalombe       Southern           12              0.5             0.8                0.5
   6  Balaka         Southern            0              0               0.8                0.5
   7  Blantyre City  Southern            0              0               0.8                0.5
   8  Chikwawa       Southern            0              0               0.8                0.5
   9  Chitipa        Northern            0              0               0.8                0.5
  10  Dedza          Central             0              0               0.8                0.5
  11  Dowa           Central             0              0               0                  0
  12  Karonga        Northern            0              0               0.8                0.5
  13  Kasungu        Central             0              0               0.8                0.5
  14  Likoma         Northern            0              0               0.8                0.5
  15  Lilongwe       Central             0              0               0.8                0.5
  16  Lilongwe City  Central             0              0               0.8                0.5
  17  Machinga       Southern            0              0               0.8                0.5
  18  Mangochi       Southern            0              0               0.8                0.5
  19  Mchinji        Central             0              0               0.8                0.5
  20  Mwanza         Southern            0              0               0.8                0.5
  21  Mzimba         Northern            0              0               0.8                0.5
  22  Mzuzu City     Northern            0              0               0.8                0.5
  23  Neno           Southern            0              0               0.8                0.5
  24  Nkhata Bay     Northern            0              0               0.8                0.5
  25  Ntcheu         Central             0              0               0.8                0.5
  26  Ntchisi        Central             0              0               0.8                0.5
  27  Rumphi         Northern            0              0               0                  0
  28  Salima         Central             0              0               0.8                0.5
  29  Thyolo         Southern            0              0               0.8                0.5
  30  Zomba          Southern            0              0               0.8                0.5
  31  Zomba City     Southern            0              0               0.8                0.5
====  =============  ========  ===========  ===============  ==============  =================

