Breast Cancer (.xlsx)
=====================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Breast_Cancer.xlsx>`

.. contents::

parameter_values
----------------

====  =================================================================================  ==================================
  ..  parameter\_name                                                                    value
====  =================================================================================  ==================================
   0  init\_prop\_breast\_cancer\_stage                                                  [0.00005,0.00005,0.00005,0.000015]
   1  init\_prop\_breast\_lump\_discernible\_breast\_cancer\_by\_stage                   [0.5,0.8,0.9,0.98]
   2  init\_prop\_with\_breast\_lump\_discernible\_diagnosed\_breast\_cancer\_by\_stage  [0.9,0.9,0.9,0.9]
   3  init\_prop\_treatment\_status\_breast\_cancer                                      [0.1,0.1,0.1,0.1]
   4  init\_prob\_palliative\_care                                                       0.8
   5  r\_stage1\_none                                                                    2e-06
   6  rr\_stage1\_none\_age3049                                                          20
   7  rr\_stage1\_none\_agege50                                                          20
   8  r\_stage2\_stage1                                                                  0.15
   9  rr\_stage2\_undergone\_curative\_treatment                                         0.05
  10  r\_stage3\_stage2                                                                  0.15
  11  rr\_stage3\_undergone\_curative\_treatment                                         0.05
  12  r\_stage4\_stage3                                                                  0.15
  13  rr\_stage4\_undergone\_curative\_treatment                                         0.5
  14  r\_death\_breast\_cancer                                                           0.1
  15  r\_breast\_lump\_discernible\_stage1                                               0.05
  16  rr\_breast\_lump\_discernible\_stage2                                              2
  17  rr\_breast\_lump\_discernible\_stage3                                              4
  18  rr\_breast\_lump\_discernible\_stage4                                              10
  19  rp\_breast\_cancer\_age3049                                                        20
  20  rp\_breast\_cancer\_agege50                                                        20
  21  sensitivity\_of\_biopsy\_for\_stage1\_breast\_cancer                               0.8
  22  sensitivity\_of\_biopsy\_for\_stage2\_breast\_cancer                               0.95
  23  sensitivity\_of\_biopsy\_for\_stage3\_breast\_cancer                               0.95
  24  sensitivity\_of\_biopsy\_for\_stage4\_breast\_cancer                               0.95
====  =================================================================================  ==================================

