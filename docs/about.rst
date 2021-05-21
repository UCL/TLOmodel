====================================
The TLO Model Documentation
====================================
The documents linked to below provide a complete technical description of the model in its current state.


Thanzi La Onse
========================
The TLO Model is a part of the `Thanzi La Onse <https://thanzi.org>`_ (Health for All) Collaboration, funded by `GCRF <https://www.newton-gcrf.org>`_ and `UKRI <https://www.ukri.org>`_ in the United Kingdom.


Overall Framework Design
========================
The overall framework is design to be self-documented. However, a brief introductory overview of the design is provided in _here_.


Core Modules
============

* `Demography <https://github.com/UCL/TLOmodel/issues>`_: determines population structure and deaths from causes not represented in the model.
* Lifestyle:
* HealthBurden
* SymptomManager: manages the onset and resolution of symptoms, including symptoms caused by conditions not included in the model.
* HealthCareSeeking: determines if and how persons seek care following the development of a symptom of ill-health.

Click :download:`here <./write-ups/Labour.docx>` to read about the Labour Module.

Contraception, Pregnancy and Labour
===================================
* `Contraception <https://github.com/UCL/TLOmodel/issues>`_:
* `Pregnancy & Labour <https://github.com/UCL/TLOmodel/issues>`_:
* `Newborn Conditions <https://github.com/UCL/TLOmodel/issues>`_:


Diseases and Other Causes of Death and Ill-Health
=================================================

Communicable Conditions
-----------------------
* HIV https://github.com/UCL/TLOmodel/blob/master/docs/write-ups/Diarrhoea.docx
* Tuberculosis (Forthcoming)
* Malaria (Forthcoming)

Diseases of Early Childhood
-----------------------
* `Diarrhoea <https://github.com/UCL/TLOmodel/blob/master/docs/write-ups/Diarrhoea.docx>`_
* Acute Lower Respiratory Infection (Forthcoming)

Non-Communicable Conditions
-----------------------

* Cancers:
    * Bladder Cancer
    * Oesophageal Cancer
* Depression
* Other non-communicable conditions:
    *


Representation of the Healthcare System
---------------------------------------

* HealthSystem: tracks the capabilities and usage and determines availability of healthcare work time, consumables and equipment for in-patient care.
* DxManager
