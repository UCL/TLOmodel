Prostate Cancer (.xlsx)
=======================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Prostate_Cancer.xlsx>`

.. contents::

parameter_values
----------------

====  ============================================================================  =======================  ============
  ..  parameter\_name                                                               value                    Unnamed: 2
====  ============================================================================  =======================  ============
   0  init\_prop\_prostate\_ca\_stage                                               [0.0001,0.0001,0.00005]
   1  init\_prop\_urinary\_symptoms\_by\_stage                                      [0.3,0.5,0.7]
   2  init\_prop\_pelvic\_pain\_symptoms\_by\_stage                                 [0.0,0.3,0.7]
   3  init\_prop\_with\_urinary\_symptoms\_diagnosed\_prostate\_ca\_by\_stage       [0.3,0.4,0.5]
   4  init\_prop\_with\_pelvic\_pain\_symptoms\_diagnosed\_prostate\_ca\_by\_stage  [0.3,0.4,0.5]
   5  init\_prop\_treatment\_status\_prostate\_ca                                   [0.1,0.1,0.1]
   6  init\_prob\_palliative\_care                                                  0.95
   7  r\_prostate\_confined\_prostate\_ca\_none                                     4e-05
   8  rr\_prostate\_confined\_prostate\_ca\_age5069                                 2
   9  rr\_prostate\_confined\_prostate\_ca\_agege70                                 4
  10  r\_local\_ln\_prostate\_ca\_prostate\_confined                                0.15
  11  rr\_local\_ln\_prostate\_ca\_undergone\_curative\_treatment                   0.1
  12  r\_metastatic\_prostate\_ca\_local\_ln                                        0.15
  13  rr\_metastatic\_prostate\_ca\_undergone\_curative\_treatment                  0.2
  14  r\_death\_metastatic\_prostate\_cancer                                        0.2
  15  r\_urinary\_symptoms\_prostate\_ca                                            0.05
  16  rr\_urinary\_symptoms\_local\_ln\_or\_metastatic\_prostate\_cancer            1.5
  17  r\_pelvic\_pain\_symptoms\_local\_ln\_prostate\_ca                            0.05
  18  rr\_pelvic\_pain\_metastatic\_prostate\_cancer                                1.5
  19  rp\_prostate\_cancer\_age5069                                                 2
  20  rp\_prostate\_cancer\_agege70                                                 4
  21  sensitivity\_of\_psa\_test\_for\_prostate\_ca                                 0.9
  22  sensitivity\_of\_biopsy\_for\_prostate\_ca                                    0.95
====  ============================================================================  =======================  ============

