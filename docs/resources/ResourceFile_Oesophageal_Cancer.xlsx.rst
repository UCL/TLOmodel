Oesophageal Cancer (.xlsx)
==========================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Oesophageal_Cancer.xlsx>`

.. contents::

parameter_values
----------------

====  ==============================================================  =====================================================
  ..  parameter\_name                                                 value
====  ==============================================================  =====================================================
   0  r\_low\_grade\_dysplasia\_none                                  5e-07
   1  rr\_low\_grade\_dysplasia\_none\_female                         1.3
   2  rr\_low\_grade\_dysplasia\_none\_per\_year\_older               1.1
   3  rr\_low\_grade\_dysplasia\_none\_tobacco                        2
   4  rr\_low\_grade\_dysplasia\_none\_ex\_alc                        1
   5  r\_high\_grade\_dysplasia\_low\_grade\_dysp                     0.03
   6  rr\_high\_grade\_dysp\_undergone\_curative\_treatment           0.1
   7  r\_stage1\_high\_grade\_dysp                                    0.01
   8  rr\_stage1\_undergone\_curative\_treatment                      0.1
   9  r\_stage2\_stage1                                               0.05
  10  rr\_stage2\_undergone\_curative\_treatment                      0.1
  11  r\_stage3\_stage2                                               0.05
  12  rr\_stage3\_undergone\_curative\_treatment                      0.1
  13  r\_stage4\_stage3                                               0.05
  14  rr\_stage4\_undergone\_curative\_treatment                      0.3
  15  r\_death\_oesoph\_cancer                                        0.4
  16  r\_dysphagia\_stage1                                            0.01
  17  rr\_dysphagia\_low\_grade\_dysp                                 1
  18  rr\_dysphagia\_high\_grade\_dysp                                1
  19  rr\_dysphagia\_stage2                                           2
  20  rr\_dysphagia\_stage3                                           50
  21  rate\_palliative\_care\_stage4                                  0.5
  22  rr\_dysphagia\_stage4                                           100
  23  init\_prop\_oes\_cancer\_stage                                  [0.0003, 0.0001, 0.000005, 0.00003, 0.00005, 0.00001]
  24  rp\_oes\_cancer\_female                                         1.3
  25  rp\_oes\_cancer\_per\_year\_older                               1.1
  26  rp\_oes\_cancer\_tobacco                                        2
  27  rp\_oes\_cancer\_ex\_alc                                        1
  28  init\_prop\_dysphagia\_oes\_cancer\_by\_stage                   [0.01, 0.03, 0.1, 0.2, 0.3, 0.8]
  29  init\_prop\_with\_dysphagia\_diagnosed\_oes\_cancer\_by\_stage  [0.8,0.85,0.9,0.95,0.98,0.9]
  30  init\_prob\_palliative\_care                                    0.5
  31  init\_prop\_treatment\_status\_oes\_cancer                      [0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
  32  sensitivity\_of\_endoscopy\_for\_oes\_cancer\_with\_dysphagia   1
====  ==============================================================  =====================================================

