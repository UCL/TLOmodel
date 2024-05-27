Bladder Cancer (.xlsx)
======================

:download:`Download original .xlsx file from GitHub <https://github.com/UCL/TLOmodel/raw/master/resources/ResourceFile_Bladder_Cancer.xlsx>`

.. contents::

parameter_values
----------------

====  =====================================================================  ===========================
  ..  parameter\_name                                                        value
====  =====================================================================  ===========================
   0  init\_prop\_bladder\_cancer\_stage                                     [0.00005, 0.00005, 0.00005]
   1  init\_prop\_blood\_urine\_bladder\_cancer\_by\_stage                   [0.20, 0.50, 0.70]
   2  init\_prop\_pelvic\_pain\_bladder\_cancer\_by\_stage                   [0.00,0.30,0.50]
   3  init\_prop\_with\_blood\_urine\_diagnosed\_bladder\_cancer\_by\_stage  [0.7,0.9,0.9]
   4  init\_prop\_with\_pelvic\_pain\_diagnosed\_bladder\_cancer\_by\_stage  [0.0,0.7,0.9]
   5  init\_prop\_treatment\_status\_bladder\_cancer                         [0.05,0.05,0.05]
   6  r\_tis\_t1\_bladder\_cancer\_none                                      1e-05
   7  rr\_tis\_t1\_bladder\_cancer\_none\_age3049                            2
   8  rr\_tis\_t1\_bladder\_cancer\_none\_age5069                            3
   9  rr\_tis\_t1\_bladder\_cancer\_none\_agege70                            3
  10  rr\_tis\_t1\_bladder\_cancer\_none\_tobacco                            3.5
  11  rr\_tis\_t1\_bladder\_cancer\_none\_schisto\_h                         5
  12  r\_t2p\_bladder\_cancer\_tis\_t1                                       0.05
  13  rr\_t2p\_bladder\_cancer\_undergone\_curative\_treatment               0.1
  14  r\_metastatic\_t2p\_bladder\_cancer                                    0.05
  15  rr\_metastatic\_undergone\_curative\_treatment                         0.2
  16  rate\_palliative\_care\_metastatic                                     0.2
  17  r\_death\_bladder\_cancer                                              0.05
  18  r\_blood\_urine\_tis\_t1\_bladder\_cancer                              0.01
  19  rr\_blood\_urine\_t2p\_bladder\_cancer                                 3
  20  rr\_blood\_urine\_metastatic\_bladder\_cancer                          3
  21  r\_pelvic\_pain\_tis\_t1\_bladder\_cancer                              0.0003
  22  rr\_pelvic\_pain\_t2p\_bladder\_cancer                                 200
  23  rr\_pelvic\_pain\_metastatic\_bladder\_cancer                          300
  24  rp\_bladder\_cancer\_age3049                                           2
  25  rp\_bladder\_cancer\_age5069                                           3
  26  rp\_bladder\_cancer\_agege70                                           3
  27  rp\_bladder\_cancer\_tobacco                                           3.5
  28  rp\_bladder\_cancer\_schisto\_h                                        5
  29  sensitivity\_of\_cystoscopy\_for\_bladder\_cancer\_blood\_urine        1
  30  sensitivity\_of\_cystoscopy\_for\_bladder\_cancer\_pelvic\_pain        1
  31  init\_prob\_palliative\_care                                           0.95
====  =====================================================================  ===========================

