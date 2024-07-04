Alri (.xlsx)
============

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Alri.xlsx>`

.. contents::

Calculations
------------

====  ============================================================  ===================  ====================  ==================  ============================================  =====================  ====================================================  ============  ====================  =====================  ======================  ======================
  ..  parameter\_name                                               1-11                 12-23                 24-59               Unnamed: 4                                    Unnamed: 5             Unnamed: 6                                            Unnamed: 7    Unnamed: 8            Unnamed: 9             Unnamed: 10             Unnamed: 11
====  ============================================================  ===================  ====================  ==================  ============================================  =====================  ====================================================  ============  ====================  =====================  ======================  ======================
   0  base\_incidence\_ALRI\_by\_agecat                             0.607                0.3429                0.1204
   1  proportion of CXR+                                            0.50475              0.5452                0.4421
   2  Target values for 12 monthly incidence rate - PNEUMONIA ONLY                                                                 1-11                                          12-23                  24-59
   3  base\_incidence\_pneumonia\_all\_pathogen                                                                                    0.30638325                                    0.18694908             0.05322884
   4                                                                                                                               Pneumonia incidence by pathogen
   5  pneumonia\_attributable\_fraction\_RSV                        0.397                0.164                 0.165               0.12163415025                                 0.03065964912          0.008782758600000001
   6  pneumonia\_attributable\_fraction\_rhinovirus                 0.029                0.152                 0.159               0.008885114250000001                          0.028416260159999998   0.00846338556
   7  pneumonia\_attributable\_fraction\_hmpv                       0.083                0.063                 0.059               0.02542980975                                 0.011777792039999999   0.00314050156
   8  pneumonia\_attributable\_fraction\_parainfluenza              0.078                0.067                 0.068               0.0238978935                                  0.01252558836          0.0036195611200000003
   9  pneumonia\_attributable\_fraction\_streptococcus              0.047                0.106                 0.092               0.01440001275                                 0.019816602479999997   0.00489705328
  10  pneumonia\_attributable\_fraction\_hib                        0.059                0.067                 0.044               0.01807661175                                 0.01252558836          0.0023420689599999997
  11  pneumonia\_attributable\_fraction\_TB                         0.067                0.039                 0.059               0.02052767775                                 0.00729101412          0.00314050156
  12  pneumonia\_attributable\_fraction\_staph                      0.037                0.009                 0.012               0.01133618025                                 0.0016825417199999998  0.00063874608
  13  pneumonia\_attributable\_fraction\_influenza                  0.016                0.029                 0.032               0.004902132                                   0.00542152332          0.0017033228800000001
  14  pneumonia\_attributable\_fraction\_jirovecii                  0.03                 0.004                 0.002               0.0091914975                                  0.0007477963199999999  0.00010645768000000001
  15  pneumonia\_attributable\_fraction\_adenovirus                 0                    0                     0                   0                                             0                      0
  16  pneumonia\_attributable\_fraction\_coronavirus                0                    0                     0                   0                                             0                      0
  17  pneumonia\_attributable\_fraction\_bocavirus                  0                    0                     0                   0                                             0                      0
  18  pneumonia\_attributable\_fraction\_other\_pathogens           0.098                0.274                 0.282               0.0300255585                                  0.05122404792          0.015010532879999998
  19  pneumonia\_attributable\_fraction\_other\_cause               0.05899999999999994  0.025999999999999912  0.0259999999999998  0.01807661174999998                           0.004860676079999984   0.0013839498399999895
  20  proportion of CXR- (bronchiolitis)                            0.49524999999999997  0.4548                N/A                 Bronchiolitis incidence by pathogen
  21  fraction due to malaria, diarrhoea, anaemia etc…              0.15                 0.15                  0.15                0.2555242375                                  0.132558282
  22  bronchiolitis\_attributable\_fraction\_RSV                    0.6                  0.6                                       0.15331454249999998                           0.0795349692
  23  bronchiolitis\_attributable\_fraction\_rhinovirus             0.05                 0.05                                      0.012776211874999999                          0.0066279141
  24  bronchiolitis\_attributable\_fraction\_hmpv                   0.05                 0.05                                      0.012776211874999999                          0.0066279141
  25  bronchiolitis\_attributable\_fraction\_parainfluenza          0.05                 0.05                                      0.012776211874999999                          0.0066279141
  26  bronchiolitis\_attributable\_fraction\_streptococcus          0                    0                                         0                                             0
  27  bronchiolitis\_attributable\_fraction\_hib                    0                    0                                         0                                             0
  28  bronchiolitis\_attributable\_fraction\_TB                     0                    0                                         0                                             0
  29  bronchiolitis\_attributable\_fraction\_staph                  0                    0                                         0                                             0
  30  bronchiolitis\_attributable\_fraction\_influenza              0.05                 0.05                                      0.012776211874999999                          0.0066279141
  31  bronchiolitis\_attributable\_fraction\_jirovecii              0                    0                                         0                                             0
  32  bronchiolitis\_attributable\_fraction\_adenovirus             0.05                 0.05                                      0.012776211874999999                          0.0066279141
  33  bronchiolitis\_attributable\_fraction\_coronavirus            0.05                 0.05                                      0.012776211874999999                          0.0066279141
  34  bronchiolitis\_attributable\_fraction\_bocavirus              0.05                 0.05                                      0.012776211874999999                          0.0066279141
  35  bronchiolitis\_attributable\_fraction\_other\_pathogens       0.05                 0.05                                      0.012776211874999999                          0.0066279141
  36                                                                                                                               0.2555242375                                  0.13255828199999994    Check if match or lower than incidence bronchiolitis
  37  Total incidence ALRI by pathogen                                                                                             Total incidence ALRI by pathogen
  38  base\_inc\_rate\_ALRI\_by\_RSV                                                                                               0.27494869275                                 0.11019461831999999    0.008782758600000001
  39  base\_inc\_rate\_ALRI\_by\_Rhinovirus                                                                                        0.021661326124999998                          0.03504417426          0.00846338556
  40  base\_inc\_rate\_ALRI\_by\_HMPV                                                                                              0.038206021625                                0.01840570614          0.00314050156
  41  base\_inc\_rate\_ALRI\_by\_Parainfluenza                                                                                     0.036674105375                                0.01915350246          0.0036195611200000003
  42  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_PCV13                                                                          0.01440001275                                 0.019816602479999997   0.00489705328
  43  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_non\_PCV13
  44  base\_inc\_rate\_ALRI\_by\_Hib
  45  base\_inc\_rate\_ALRI\_by\_H.influenzae\_non\_type\_b                                                                        0.01807661175                                 0.01252558836          0.0023420689599999997
  46  base\_inc\_rate\_ALRI\_by\_TB                                                                                                0.02052767775                                 0.00729101412          0.00314050156
  47  base\_inc\_rate\_ALRI\_by\_Staph\_aureus                                                                                     0.01133618025                                 0.0016825417199999998  0.00063874608
  48  base\_inc\_rate\_ALRI\_by\_Enterobacteriaceae
  49  base\_inc\_rate\_ALRI\_by\_other\_Strepto\_Enterococci
  50  base\_inc\_rate\_ALRI\_by\_Influenza                                                                                         0.017678343875                                0.01204943742          0.0017033228800000001
  51  base\_inc\_rate\_ALRI\_by\_P.jirovecii                                                                                       0.0091914975                                  0.0007477963199999999  0.00010645768000000001
  52  base\_inc\_rate\_ALRI\_by\_Bocavirus                                                                                         0.012776211874999999                          0.0066279141           0
  53  base\_inc\_rate\_ALRI\_by\_Adenovirus                                                                                        0.012776211874999999                          0.0066279141           0
  54  base\_inc\_rate\_ALRI\_by\_other\_viral\_pathogens
  55  base\_inc\_rate\_ALRI\_by\_other\_bacterial\_pathogens                                                                       0.042801770375                                0.05785196202          0.015010532879999998
  56
  57
  58  ALRI\_attributable\_fraction\_RSV                                                                                            [0.27494869275,0.11019461832,0.0087827586]
  59  ALRI\_attributable\_fraction\_rhinovirus                                                                                     [0.021661326125,0.03504417426,0.00846338556]
  60  ALRI\_attributable\_fraction\_hmpv                                                                                           [0.038206021625,0.01840570614,0.00314050156]
  61  ALRI\_attributable\_fraction\_parainfluenza                                                                                  [0.036674105375,0.01915350246,0.00361956112]
  62  ALRI\_attributable\_fraction\_streptococcus                                                                                  [0.01440001275,0.01981660248,0.00489705328]
  63  ALRI\_attributable\_fraction\_hib                                                                                            [0.01807661175,0.01252558836,0.00234206896]
  64  ALRI\_attributable\_fraction\_TB                                                                                             [0.02052767775,0.00729101412,0.00314050156]
  65  ALRI\_attributable\_fraction\_staph                                                                                          [0.01133618025,0.00168254172,0.00063874608]
  66  ALRI\_attributable\_fraction\_influenza                                                                                      [0.017678343875,0.01204943742,0.00170332288]
  67  ALRI\_attributable\_fraction\_jirovecii                                                                                      [0.0091914975,0.00074779632,0.00010645768]
  68  ALRI\_attributable\_fraction\_adenovirus                                                                                     [0.012776211875,0.0066279141,0]
  69  ALRI\_attributable\_fraction\_coronavirus                                                                                    [,,]
  70  ALRI\_attributable\_fraction\_bocavirus                                                                                      [0.012776211875,0.0066279141,0]
  71  ALRI\_attributable\_fraction\_other\_pathogens                                                                               [0.042801770375,0.05785196202,0.01501053288]
  72
  73  Total incidence ALRI by pathogen                                                                                             Total incidence ALRI by pathogen
  74  base\_inc\_rate\_ALRI\_by\_RSV                                                                                               0.27494869275                                 0.11019461832          0.008782758600000001                                                0.27494869275         0.11019461832          0.008782758600000001    0.2749, 0.1102, 0.0088
  75  base\_inc\_rate\_ALRI\_by\_Rhinovirus                                                                                        0.021661326124999998                          0.03504417426          0.00846338556                                                       0.021661326124999998  0.03504417426          0.00846338556           0.0217, 0.035, 0.0085
  76  base\_inc\_rate\_ALRI\_by\_HMPV                                                                                              0.038206021625                                0.01840570614          0.00314050156                                                       0.038206021625        0.01840570614          0.00314050156           0.0382, 0.0184, 0.0031
  77  base\_inc\_rate\_ALRI\_by\_Parainfluenza                                                                                     0.036674105375                                0.01915350246          0.0036195611200000003                                               0.036674105375        0.01915350246          0.0036195611200000003   0.0367, 0.0192, 0.0036
  78  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_PCV13                                                                          0.007200006375                                0.009908301239999999   0.00244852664                                                       0.007200006375        0.009908301239999999   0.00244852664           0.0072, 0.0099, 0.0024
  79  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_non\_PCV13                                                                     0.007200006375                                0.009908301239999999   0.00244852664                                                       0.007200006375        0.009908301239999999   0.00244852664           0.0072, 0.0099, 0.0024
  80  base\_inc\_rate\_ALRI\_by\_Hib                                                                                               0.009038305875                                0.00626279418          0.0011710344799999999                                               0.009038305875        0.00626279418          0.0011710344799999999   0.009, 0.0063, 0.0012
  81  base\_inc\_rate\_ALRI\_by\_H.influenzae\_non\_type\_b                                                                        0.009038305875                                0.00626279418          0.0011710344799999999                                               0.009038305875        0.00626279418          0.0011710344799999999   0.009, 0.0063, 0.0012
  82  base\_inc\_rate\_ALRI\_by\_TB                                                                                                0.02052767775                                 0.00729101412          0.00314050156                                                       0.02052767775         0.00729101412          0.00314050156           0.0205, 0.0073, 0.0031
  83  base\_inc\_rate\_ALRI\_by\_Staph\_aureus                                                                                     0.01133618025                                 0.0016825417199999998  0.00063874608                                                       0.01133618025         0.0016825417199999998  0.00063874608           0.0113, 0.0017, 0.0006
  84  base\_inc\_rate\_ALRI\_by\_Enterobacteriaceae                                                                                0.010263838875                                0.00364550706          0.00157025078                                                       0.010263838875        0.00364550706          0.00157025078           0.0103, 0.0036, 0.0016
  85  base\_inc\_rate\_ALRI\_by\_other\_Strepto\_Enterococci                                                                       0.010263838875                                0.00364550706          0.00157025078                                                       0.010263838875        0.00364550706          0.00157025078           0.0103, 0.0036, 0.0016
  86  base\_inc\_rate\_ALRI\_by\_Influenza                                                                                         0.017678343875                                0.01204943742          0.0017033228800000001                                               0.017678343875        0.01204943742          0.0017033228800000001   0.0177, 0.012, 0.0017
  87  base\_inc\_rate\_ALRI\_by\_P.jirovecii                                                                                       0.0091914975                                  0.0007477963199999999  0.00010645768000000001                                              0.0091914975          0.0007477963199999999  0.00010645768000000001  0.0092, 0.0007, 0.0001
  88  base\_inc\_rate\_ALRI\_by\_Bocavirus                                                                                         0.012776211874999999                          0.0066279141           0                                                                   0.012776211874999999  0.0066279141           0                       0.0128, 0.0066, 0
  89  base\_inc\_rate\_ALRI\_by\_Adenovirus                                                                                        0.012776211874999999                          0.0066279141           0                                                                   0.012776211874999999  0.0066279141           0                       0.0128, 0.0066, 0
  90  base\_inc\_rate\_ALRI\_by\_other\_viral\_pathogens                                                                           0.0001                                        0.0001                 0.0001                                                              0.0001                0.0001                 0.0001                  0.0001, 0.0001, 0.0001
  91  base\_inc\_rate\_ALRI\_by\_other\_bacterial\_pathogens                                                                       0.042801770375                                0.05785196202          0.015010532879999998                                                0.042801770375        0.05785196202          0.015010532879999998    0.0428, 0.0579, 0.015
  92
  93
  94  base\_inc\_rate\_ALRI\_by\_RSV                                                                                               [0.27494869275,0.11019461832,0.0087827586]
  95  base\_inc\_rate\_ALRI\_by\_Rhinovirus                                                                                        [0.021661326125,0.03504417426,0.00846338556]
  96  base\_inc\_rate\_ALRI\_by\_HMPV                                                                                              [0.038206021625,0.01840570614,0.00314050156]
  97  base\_inc\_rate\_ALRI\_by\_Parainfluenza                                                                                     [0.036674105375,0.01915350246,0.00361956112]
  98  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_PCV13                                                                          [0.007200006375,0.00990830124,0.00244852664]
  99  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_non\_PCV13                                                                     [0.007200006375,0.00990830124,0.00244852664]
 100  base\_inc\_rate\_ALRI\_by\_Hib                                                                                               [0.009038305875,0.00626279418,0.00117103448]
 101  base\_inc\_rate\_ALRI\_by\_H.influenzae\_non\_type\_b                                                                        [0.009038305875,0.00626279418,0.00117103448]
 102  base\_inc\_rate\_ALRI\_by\_Staph\_aureus                                                                                     [0.01133618025,0.00168254172,0.00063874608]
 103  base\_inc\_rate\_ALRI\_by\_Enterobacteriaceae                                                                                [0.010263838875,0.00364550706,0.00157025078]
 104  base\_inc\_rate\_ALRI\_by\_other\_Strepto\_Enterococci                                                                       [0.010263838875,0.00364550706,0.00157025078]
 105  base\_inc\_rate\_ALRI\_by\_Influenza                                                                                         [0.017678343875,0.01204943742,0.00170332288]
 106  base\_inc\_rate\_ALRI\_by\_P.jirovecii                                                                                       [0.0091914975,0.00074779632,0.00010645768]
 107  base\_inc\_rate\_ALRI\_by\_Bocavirus                                                                                         [0.012776211875,0.0066279141,0]
 108  base\_inc\_rate\_ALRI\_by\_Adenovirus                                                                                        [0.012776211875,0.0066279141,0]
 109  base\_inc\_rate\_ALRI\_by\_other\_viral\_pathogens                                                                           [0.0001,0.0001,0.0001]
 110  base\_inc\_rate\_ALRI\_by\_other\_bacterial\_pathogens                                                                       [0.042801770375,0.05785196202,0.01501053288]
====  ============================================================  ===================  ====================  ==================  ============================================  =====================  ====================================================  ============  ====================  =====================  ======================  ======================

Parameter_values
----------------

====  =================================================================================  ===============================================================  =====================================================  ===========================================================================================  ==================================================================================================================================================================================  =====================
  ..  parameter\_name                                                                    value                                                            source                                                 notes                                                                                        Unnamed: 4                                                                                                                                                                          Unnamed: 5
====  =================================================================================  ===============================================================  =====================================================  ===========================================================================================  ==================================================================================================================================================================================  =====================
   0  base\_inc\_rate\_ALRI\_by\_RSV                                                     [0.165623001012122,0.0368943809605977,0.011883807410198]
   1  base\_inc\_rate\_ALRI\_by\_Rhinovirus                                              [0.0243368081563306,0.0564272209432261,0.0181754025659935]
   2  base\_inc\_rate\_ALRI\_by\_HMPV                                                    [0.0316242920447732,0.01106932885584,0.00356546901880363]
   3  base\_inc\_rate\_ALRI\_by\_Parainfluenza                                           [0.0400689053941642,0.0125990679490263,0.00405820325903094]
   4  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_PCV13                                [0.00839805492638728,0.0086009815601405,0.00277040583791159]
   5  base\_inc\_rate\_ALRI\_by\_Strep\_pneumoniae\_non\_PCV13                           [0.00971808948465436,0.0056295094957977,0.00181328443302487]
   6  base\_inc\_rate\_ALRI\_by\_Hib                                                     [0.00456369209787862,0.00223809928508262,0.000720899502209533]
   7  base\_inc\_rate\_ALRI\_by\_H.influenzae\_non\_type\_b                              [0.0178046628079453,0.00771804577106646,0.00248600917371134]
   8  base\_inc\_rate\_ALRI\_by\_Staph\_aureus                                           [0.0230583713892271,0.00164959730828988,0.000531340984878809]
   9  base\_inc\_rate\_ALRI\_by\_Enterobacteriaceae                                      [0.00577447898013911,0.00776959582619209,0.002502613624339]
  10  base\_inc\_rate\_ALRI\_by\_other\_Strepto\_Enterococci                             [0.00739221008020393,0.000441075977443103,0.000142072094251877]
  11  base\_inc\_rate\_ALRI\_by\_Influenza                                               [0.0156895896609387,0.00577122651203705,0.00185893197294524]
  12  base\_inc\_rate\_ALRI\_by\_P.jirovecii                                             [0.00532021183938121,0.00406616141018573,0.00130972462037058]
  13  base\_inc\_rate\_ALRI\_by\_other\_viral\_pathogens                                 [0.0289052989892286,0.0340001707821008,0.0109515723253317]
  14  base\_inc\_rate\_ALRI\_by\_other\_bacterial\_pathogens                             [0.0340967151177136,0.0275272316777052,0.00886661630515269]
  15  base\_inc\_rate\_ALRI\_by\_other\_pathogens\_NoS                                   [0.012096954443426,0.0244908799021158,0.00788859692140824]
  16  proportion\_pneumonia\_in\_RSV\_ALRI                                               [0.469523118748083,0.493147834326293]
  17  proportion\_pneumonia\_in\_Rhinovirus\_ALRI                                        [0.233411098940226,0.300943850285401]
  18  proportion\_pneumonia\_in\_HMPV\_ALRI                                              [0.514096212711081,0.607661832402657]
  19  proportion\_pneumonia\_in\_Parainfluenza\_ALRI                                     [0.381306534835401,0.586394378700289]
  20  proportion\_pneumonia\_in\_Strep\_pneumoniae\_PCV13\_ALRI                          [0.419837201436785,1]
  21  proportion\_pneumonia\_in\_Strep\_pneumoniae\_non\_PCV13\_ALRI                     [0.584526531211357,0.47010416567094]
  22  proportion\_pneumonia\_in\_Hib\_ALRI                                               [0.643816417750475,0.591228432603006]
  23  proportion\_pneumonia\_in\_H.influenzae\_non\_type\_b\_ALRI                        [0.484067635967411,0.685783925920651]
  24  proportion\_pneumonia\_in\_Staph\_aureus\_ALRI                                     [0.314311461687961,0.668460076932899]
  25  proportion\_pneumonia\_in\_Enterobacteriaceae\_ALRI                                [0.542743088827561,0.397386416381317]
  26  proportion\_pneumonia\_in\_other\_Strepto\_Enterococci\_ALRI                       [0.794939501527616,1]
  27  proportion\_pneumonia\_in\_Influenza\_ALRI                                         [0.274661773216931,0.477667070122735]
  28  proportion\_pneumonia\_in\_P.jirovecii\_ALRI                                       [0.773174448993322,0.650848699225084]
  29  proportion\_pneumonia\_in\_other\_viral\_pathogens\_ALRI                           [0.264286065267648,0.301616616612727]
  30  proportion\_pneumonia\_in\_other\_bacterial\_pathogens\_ALRI                       [0.321708555070697,0.424616377668989]
  31  proportion\_pneumonia\_in\_other\_pathogens\_NoS\_ALRI                             [0.323847892119005,0.256639724815326]
  32
  33  rr\_ALRI\_HIV/AIDS                                                                 6.51                                                             Jackson et al. 2013
  34  rr\_ALRI\_incomplete\_measles\_immunisation                                        1.8                                                              Jackson et al. 2013
  35  rr\_ALRI\_low\_birth\_weight                                                       3.6                                                              Jackson et al. 2013
  36  rr\_ALRI\_non\_exclusive\_breastfeeding                                            2.7                                                              Jackson et al. 2013
  37  rr\_ALRI\_indoor\_air\_pollution                                                   1.57                                                             Jackson et al. 2013
  38
  39  rr\_Strep\_pneum\_VT\_ALRI\_with\_PCV13\_age<2y                                    0.18                                                             Mackenzie et al. 2016 Lancet
  40  rr\_Strep\_pneum\_VT\_ALRI\_with\_PCV13\_age2to5y                                  0.32                                                             Mackenzie et al. 2016 Lancet
  41  rr\_all\_strains\_Strep\_pneum\_ALRI\_with\_PCV13                                  0.45                                                             Mackenzie et al. 2016 Lancet                           for calibration
  42  effectiveness\_Hib\_vaccine\_on\_Hib\_strains                                      0.86                                                             Obonyo and Lau 2006
  43  rr\_Hib\_ALRI\_with\_Hib\_vaccine                                                  0.14
  44  prob\_viral\_pneumonia\_bacterial\_coinfection                                     0.3
  45  proportion\_bacterial\_coinfection\_pathogen                                       [0.107,0.072,0.038,0.141,0.134,0.117,0.052,0.338]
  46  overall\_progression\_to\_severe\_ALRI                                             0.12                                                             Walker et al. 2013                                     for calibration
  47  prob\_pulmonary\_complications\_in\_pneumonia                                      0.215                                                            Resti et al. 2010
  48  prob\_pleural\_effusion\_in\_pulmonary\_complicated\_pneumonia                     0.938                                                            Resti et al. 2010
  49  prob\_empyema\_in\_pulmonary\_complicated\_pneumonia                               0.315                                                            Resti et al. 2010
  50  prob\_lung\_abscess\_in\_pulmonary\_complicated\_pneumonia                         0.049                                                            Resti et al. 2010, assumed from necrotising pneumonia
  51  prob\_pneumothorax\_in\_pulmonary\_complicated\_pneumonia                          0.049                                                            Resti et al. 2010, assumed from atelectasia
  52  prob\_hypoxaemia\_in\_pneumonia                                                    0.479                                                            Fancourt et al 2017
  53  prob\_hypoxaemia\_in\_other\_alri                                                  0.273                                                            Fancourt et al 2017
  54  prob\_bacteraemia\_in\_pneumonia                                                   0.039                                                            Sebavonge et al. 2016
  55  prob\_progression\_to\_sepsis\_with\_bacteraemia                                   0.3                                                              Fritz et al. 2019
  56  prob\_cough\_in\_pneumonia                                                         1                                                                Rees et al 2020                                        0.77
  57  prob\_difficult\_breathing\_in\_pneumonia                                          1                                                                Rees et al 2020                                        0.7
  58  prob\_fever\_in\_pneumonia                                                         0.82                                                             Rees et al 2020
  59  prob\_chest\_indrawing\_in\_pneumonia                                              0.74                                                             Rees et al 2020
  60  prob\_tachypnoea\_in\_pneumonia                                                    0.834                                                            Fancourt et al 2017
  61  prob\_danger\_signs\_in\_pneumonia                                                 0.28                                                             Rees et al 2020
  62  prob\_cyanosis\_in\_pneumonia                                                      0.06                                                             Rees et al 2020
  63  prob\_cough\_in\_other\_alri                                                       1                                                                assumed
  64  prob\_difficult\_breathing\_in\_other\_alri                                        1                                                                assumed
  65  prob\_fever\_in\_other\_alri                                                       0.76                                                                                                                    calculated using OR=1.41 temp>38C from Fancourt and the prob\_fever\_in\_pneumonia
  66  prob\_chest\_indrawing\_in\_other\_alri                                            0.33                                                                                                                    assumed (based on Eric' Bangladesh trial - outpatient pneumonia)
  67  prob\_tachypnoea\_in\_other\_alri                                                  0.758                                                            Fancourt et al 2017
  68  prob\_danger\_signs\_in\_other\_alri                                               0.3                                                              Rees et al 2020                                        1-specificity                                                                                or 0.03 assumed (based on Eric' Bangladesh trial - outpatient pneumonia)
  69  prob\_cyanosis\_in\_other\_alri                                                    0.04                                                             Rees et al 2020                                        1-specificity
  70  prob\_danger\_signs\_in\_sepsis                                                    1                                                                assumed
  71  prob\_danger\_signs\_in\_SpO2<90%                                                  0.313                                                            McCollum et al. 2016                                   HC only
  72  prob\_danger\_signs\_in\_SpO2\_90-92%                                              0.133                                                            McCollum et al. 2016                                   HC only
  73  prob\_chest\_indrawing\_in\_SpO2<90%                                               0.701                                                            McCollum et al. 2016                                   HC & CHW combined
  74  prob\_chest\_indrawing\_in\_SpO2\_90-92%                                           0.445                                                            McCollum et al. 2016                                   HC & CHW combined
  75  proportion\_hypoxaemia\_with\_SpO2<90%                                             0.3576                                                           McCollum et al. 2016                                   prevalence of SpO2 <90% over total prevalence of hypoxaemia
  76  prob\_cyanosis\_in\_SpO2<90%                                                       0.078                                                            Bassat et al 2016                                      cyanosis in Hypoxaemic children < 90% in Mozambique study. Non-hypoxaemic children had 1.5%
  77
  78  max\_alri\_duration\_in\_days\_without\_treatment                                  14
  79  days\_between\_treatment\_and\_cure                                                28                                                                                                                      7 inpatient days + 3 weeks oral antibiotics course
  80
  81  tf\_3day\_amoxicillin\_for\_fast\_breathing\_with\_SpO2>=90%                       0.101                                                            Ginsburg et al. 2019                                   treatment failure or relapse by day 14
  82  tf\_5day\_amoxicillin\_for\_chest\_indrawing\_with\_SpO2>=90%                      0.108                                                            Ginsburg et al. 2020                                   treatment failure or relapse by day 14                                                       5-day course of oral amoxicillin treatment failure of non-hypoxaemic (>=90%) chest-indrawing pneumonia --- 4.3% --- check with ref. EMPIC Study Group (Malawi smaller sample size)
  83  tf\_3day\_amoxicillin\_for\_chest\_indrawing\_with\_SpO2>=90%                      0.125                                                            Ginsburg et al. 2020                                   treatment failure or relapse by day 15
  84  tf\_7day\_amoxicillin\_for\_fast\_breathing\_pneumonia\_in\_young\_infants         0.054                                                                                                                   EMPIC study 2021, 6.3% TF by standard 1st dose antibiotic and referral
  85  tf\_oral\_amoxicillin\_only\_for\_severe\_pneumonia\_with\_SpO2>=90%               0.6
  86  tf\_oral\_amoxicillin\_only\_for\_non\_severe\_pneumonia\_with\_SpO2<90%           0.2826                                                                                                                  calculated using 0.17 and pOR=0.52
  87  tf\_oral\_amoxicillin\_only\_for\_severe\_pneumonia\_with\_SpO2<90%                0.8
  88
  89  sensitivity\_of\_classification\_of\_fast\_breathing\_pneumonia\_facility\_level0  0.7                                                              HHFA 2018                                                                                                                                           0.746                                                                                                                                                                               Boyce et al 2019
  90  sensitivity\_of\_classification\_of\_danger\_signs\_pneumonia\_facility\_level0    0.7                                                              HHFA 2018                                                                                                                                           0.708                                                                                                                                                                               Boyce et al 2019
  91  sensitivity\_of\_classification\_of\_non\_severe\_pneumonia\_facility\_level1      0.7                                                              HHFA 2018                                                                                                                                           0.3                                                                                                                                                                                 Bjornstad et al 2014
  92  sensitivity\_of\_classification\_of\_severe\_pneumonia\_facility\_level1           0.7                                                              HHFA 2018                                                                                                                                           0.0592                                                                                                                                                                              Bjornstad et al 2014
  93  sensitivity\_of\_classification\_of\_non\_severe\_pneumonia\_facility\_level2      0.7                                                              HHFA 2018                                                                                                                                           0.23                                                                                                                                                                                Uwemedimo et al. 2018
  94  sensitivity\_of\_classification\_of\_severe\_pneumonia\_facility\_level2           0.7                                                              HHFA 2018
  95  prob\_iCCM\_severe\_pneumonia\_treated\_as\_fast\_breathing\_pneumonia             0.7                                                              dummy
  96  prob\_IMCI\_severe\_pneumonia\_treated\_as\_non\_severe\_pneumonia                 0.7                                                              dummy
  97
  98  scaler\_on\_risk\_of\_death                                                        0.75                                                             For Calibration
  99  base\_odds\_death\_ALRI\_age<2mo                                                   0.041965                                                         Lazzerini et al 2016
 100  or\_death\_ALRI\_age<2mo\_very\_severe\_pneumonia                                  3.994783                                                         Lazzerini et al 2016
 101  or\_death\_ALRI\_age<2mo\_P.jirovecii                                              5.460986                                                         Lazzerini et al 2016
 102  or\_death\_ALRI\_age<2mo\_by\_month\_increase\_in\_age                             0.6033254                                                        Lazzerini et al 2016
 103
 104  base\_odds\_death\_ALRI\_age2\_59mo                                                0.0244725                                                        Lazzerini et al 2016                                   calculated
 105  or\_death\_ALRI\_age2\_59mo\_very\_severe\_pneumonia                               8.382832                                                         Lazzerini et al 2016
 106  or\_death\_ALRI\_age2\_59mo\_P.jirovecii                                           13.74696                                                         Lazzerini et al 2016
 107  or\_death\_ALRI\_age2\_59mo\_SAM                                                   7.216448                                                         Lazzerini et al 2016
 108  or\_death\_ALRI\_age2\_59mo\_by\_month\_increase\_in\_age                          0.9661072                                                        Lazzerini et al 2016
 109  or\_death\_ALRI\_age2\_59mo\_female                                                1.327285                                                         Lazzerini et al 2016
 110  or\_death\_ALRI\_SpO2<90%                                                          5.04                                                             Hooli et al. 2016
 111  or\_death\_ALRI\_SpO2\_90\_92%                                                     1.54                                                             Hooli et al. 2016
 112  or\_death\_ALRI\_sepsis                                                            151.9
 113  or\_death\_ALRI\_pneumothorax                                                      77.4
 114
 115  tf\_1st\_line\_antibiotic\_for\_severe\_pneumonia                                  0.11                                                             Lassi et al 2013                                       0.171 overall value in source
 116  rr\_tf\_1st\_line\_antibiotics\_if\_cyanosis                                       1.55                                                             Muro et al 2020
 117  rr\_tf\_1st\_line\_antibiotics\_if\_SpO2<90%                                       1.28                                                             Muro et al 2020
 118  rr\_tf\_1st\_line\_antibiotics\_if\_abnormal\_CXR                                  1.71                                                             Muro et al 2020
 119  rr\_tf\_1st\_line\_antibiotics\_if\_HIV/AIDS                                       1.8                                                              Muro et al 2020
 120  rr\_tf\_1st\_line\_antibiotics\_if\_MAM                                            1.48                                                             Muro et al 2020
 121  rr\_tf\_1st\_line\_antibiotics\_if\_SAM                                            2.02                                                             Muro et al 2020
 122  or\_mortality\_improved\_oxygen\_systems                                           0.52                                                             Lam et al 2021
 123
 124  pulse\_oximeter\_and\_oxygen\_is\_available                                        Default
 125
 126  or\_care\_seeking\_perceived\_severe\_illness                                      2.4                                                              Lungu et al 2020                                       perceived severe illness, OR used for seeking care for chest-indrawing
 127
 128  tf\_2nd\_line\_antibiotic\_for\_severe\_pneumonia                                  0.235
 129
 130  prob\_for\_followup\_if\_treatment\_failure                                        0.3
====  =================================================================================  ===============================================================  =====================================================  ===========================================================================================  ==================================================================================================================================================================================  =====================

GBD_Malawi_estimates
--------------------

====  ======  ==========================  =======================  =======================  =============================  ==========================  ==========================  =======  ==============  ==============
  ..    Year    Death\_per100k\_children    Death\_per100k\_lower    Death\_per100k\_upper    Incidence\_per100\_children    Incidence\_per100\_lower    Incidence\_per100\_upper    DALYs    DALYs\_lower    DALYs\_upper
====  ======  ==========================  =======================  =======================  =============================  ==========================  ==========================  =======  ==============  ==============
   0    2010                      258.35                   194.1                    330.88                       13.1873                     10.7688                      15.9341   567888          427703          727014
   1    2011                      257.73                   193.89                   332.68                       12.6826                     10.3766                      15.2886   570661          429499          735642
   2    2012                      246.35                   185.74                   318.6                        12.1102                      9.89684                     14.6214   549033          413888          708913
   3    2013                      232.42                   172.61                   306.36                       11.5172                      9.42606                     13.8967   520890          387278          688319
   4    2014                      220.58                   164.09                   291.38                       10.9511                      8.84937                     13.3138   475402          353236          629467
   5    2015                      210.56                   156.42                   278.84                       10.4595                      8.44016                     12.8091   496503          369177          655153
   6    2016                      195.02                   141.1                    258.3                        10.0094                      8.06142                     12.3941   441064          319310          583925
   7    2017                      176.39                   127.85                   236.58                        9.68916                     7.72748                     12.1848   399055          289917          535254
   8    2018                      169.27                   120.13                   232.52                        9.52978                     7.56549                     12.0005   383027          271830          526008
   9    2019                      158.72                   111.41                   220.28                        9.44652                     7.44754                     11.9048   360719          253591          500057
====  ======  ==========================  =======================  =======================  =============================  ==========================  ==========================  =======  ==============  ==============

McAllister_2019
---------------

====  ======  =============================  ==========================  ==========================  ============================
  ..    Year  Incidence\_per100\_children    Incidence\_per100\_lower    Incidence\_per100\_upper      Death\_per1000\_livebirths
====  ======  =============================  ==========================  ==========================  ============================
   0    2000  39.6                           23.5                        66.9                                                28.2
   1    2005                                                                                                                 15.6
   2    2010                                                                                                                 11.7
   3    2015  22.1                           13.2                        37.5                                                 8.6
====  ======  =============================  ==========================  ==========================  ============================

Lazzerini CFR
-------------

====  ======  =====  ============  ============  ============  =====================
  ..  Year    CFR    CFR\_lower    CFR\_upper    Unnamed: 4    Unnamed: 5
====  ======  =====  ============  ============  ============  =====================
   0  2001    15.2   13.3          17.3
   1  2002    11.3   10.2          12.6
   2  2003    9.1    8.2           10
   3  2004    9.2    8.3           10.2
   4  2005    9.7    9             10.5
   5  2006    7.4    6.8           8
   6  2007    6.8    6.4           7.4
   7  2008    7.3    6.8           7.9
   8  2009    6.1    5.7           6.5
   9  2010    5.3    4.9           5.6
  10  2011    4.1    3.8           4.4
  11  2012    4.5    4.2           4.9
  12                                                           note: In-hospital CFR
====  ======  =====  ============  ============  ============  =====================

Pathogen_specific
-----------------

====  ======  ========================  ===============================  ===============================  ==============================  =====================================  =====================================  ===============================================================================================================================================================================================================================  ================================  ================================  ==================================  =========================================  =========================================
  ..  Year    Inc\_RSV\_ALRI\_per100    Inc\_RSV\_ALRI\_per100\_lower    Inc\_RSV\_ALRI\_per100\_upper    Inc\_Influenza\_ALRI\_per100    Inc\_Influenza\_ALRI\_per100\_lower    Inc\_Influenza\_ALRI\_per100\_upper    Inc\_HMPV\_ALRI\_per100                                                                                                                                                                                                          Inc\_HMPV\_ALRI\_per100\_lower    Inc\_HMPV\_ALRI\_per100\_upper    Inc\_Parainfluenza\_ALRI\_per100    Inc\_Parainfluenza\_ALRI\_per100\_lower    Inc\_Parainfluenza\_ALRI\_per100\_upper
====  ======  ========================  ===============================  ===============================  ==============================  =====================================  =====================================  ===============================================================================================================================================================================================================================  ================================  ================================  ==================================  =========================================  =========================================
   0
   1  2015    5.89                      3.72                             9.26
   2  2018                                                                                                1.46                            0.92                                   2.33                                   2.13                                                                                                                                                                                                                             1.55                              2.93                              3.77                                2.78                                       5.1
   3
   4                                                                                                                                                                                                                    RSV ref (Malawi-specific) : Global, regional, and national disease burden estimates of acute lower respiratory infections due to respiratory syncytial virus in young children in 2015: a systematic review and modelling study
   5                                                                                                                                                                                                                    Influenza ref (LMIC) : Global burden of respiratory infections associated with seasonal influenza in children under 5 years in 2018: a systematic review and modelling study
   6                                                                                                                                                                                                                    HMPV ref (LMIC) : Global burden of acute lower respiratory infection associated with human metapneumovirus in children under 5 years in 2018: a systematic review and modelling study
   7                                                                                                                                                                                                                    Parainfluenza ref (LMIC): Global burden of acute lower respiratory infection associated with human parainfluenza virus in children younger than 5 years for 2018: a systematic review and meta-analysis
====  ======  ========================  ===============================  ===============================  ==============================  =====================================  =====================================  ===============================================================================================================================================================================================================================  ================================  ================================  ==================================  =========================================  =========================================

