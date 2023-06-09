
====================
Model Documentation
====================

This page provides links to "user-friendly" write-ups of all the major modules that comprise the TLO Model.
In addition,

* A high-level overview of the design of the model is provided here: :download:`.docx <./write-ups/Framework.docx>`;

* The entire code is fully documented under :doc:`Reference <reference/index>`.

* A list of the health system interaction events defined in the model is provided :doc:`here <hsi_events>`.


Core Functions
===============
* **Demography**: Determines population structure and deaths from causes not represented in the model. :download:`.docx <./write-ups/Demography.docx>`

* **Lifestyle**: Determines key characteristics and risk factors that may change during the life-course (including, education, diet, smoking, alcohol consumption, access to hand-washing and sanitation). :download:`.docx <./write-ups/LifeStyle.docx>`

* **SymptomManager**: Manages the onset and resolution of symptoms, including symptoms caused by conditions not included in the model. :download:`.docx <./write-ups/SymptomManager.docx>`

* **HealthSeekingBehaviour**: Determines if and how persons seek care following the development of a symptom of ill-health. :download:`.docx <./write-ups/HealthSeekingBehaviour.docx>`

* **HealthBurden**: Tracks the occurrence of Disability-Adjusted Life-Years in the population. :download:`.docx <./write-ups/HealthBurden.docx>`


Representation of the Healthcare System
========================================
* **HealthSystem**: Tracks the availability and usage of the resources of the healthcare system with respect to healthcare worker time, consumables and equipment for in-patient care. :download:`.docx <./write-ups/HealthSystem.docx>`

* **Routine Immunization**: The services that deliver a set of immunizations to children. :download:`.docx <./write-ups/EPI.docx>`


Contraception, Pregnancy and Labour
===================================
* **Contraception**: Determines fecundity, the usage of contraception (including switching between contraceptives) and the onset of pregnancy. :download:`.docx <./write-ups/Contraception.docx>`

* **Pregnancy** Represents the antenatal period of pregnancy (the period from conception to the termination of pregnancy), including complication experienced and emergency care that may be provided. :download:`.docx <./write-ups/PregnancySupervisor.docx>`

* **Care of Women During Pregnancy**: Determines the routine care provided during pregnancy. :download:`.docx <./write-ups/CareOfWomenDuringPregnancy.docx>`

* **Labour**: Represents the labour, birth and the immediate postnatal period, including complications experienced, and the care provided (including 'skilled birth attendance' at basic or comprehensive level emergency obstetric care facilities). :download:`.docx <./write-ups/Labour.docx>`

* **Newborns** Represents the key conditions/complications experienced by a neonate and the treatments associated. :download:`.docx <./write-ups/NewbornOutcomes.docx>`

* **Postnatal Women** Represents the key conditions/complications experienced by a mother (and by association, a neonate) in the period immediately postpartum and the treatments associated. :download:`.docx <./write-ups/PostnatalSupervisor.docx>`


Conditions of Early Childhood
==============================
* **Acute Lower Respiratory Infection**: Childhood viral pneumonia, bacterial pneumonia and viral bronchiolitis and the treatments associated with each. :download:`.docx <./write-ups/Alri.docx>`

* **Diarrhoea**: Childhood diarrhoea caused by virus or bacteria resulting in dehydration, and the treatments associated. :download:`.docx <./write-ups/Diarrhoea.docx>`

* **Childhood Undernutrition**: Acute and chronic undernutrition and its effects of Wasting and Stunting. :download:`.docx <./write-ups/ChildhoodUndernutrition.docx>`


Communicable Diseases
========================
* **HIV**: HIV/AIDS and associated prevention and treatment programmes. :download:`.docx <./write-ups/Hiv.docx>`

* **TB**: Tuberculosis and associated prevention and treatment programmes. :download:`.docx <./write-ups/TB.docx>`

* **Malaria**: Malaria disease and associated prevention and treatment programmes. :download:`.docx <./write-ups/Malaria.docx>`

* **Measles**: Measles-related disease and associated prevention and treatment programmes. :download:`.docx <./write-ups/Measles.docx>`

* **Schistosomiasis**: Schistosomiasis disease and associated prevention and treatment programmes. :download:`.docx <./write-ups/Schistosomiasis.docx>`


Non-Communicable Conditions
==============================
* Cancers:
    * **Bladder Cancer**: Cancer of the bladder and its treatment. :download:`.docx <./write-ups/BladderCancer.docx>`

    * **Breast Cancer**: Cancer of the breast and its treatment. :download:`.docx <./write-ups/BreastCancer.docx>`

    * **Oesophageal Cancer**: Cancer of the oesophagus and its treatment. :download:`.docx <./write-ups/OesophagealCancer.docx>`

    * **Other Adult Cancer**: Summary representation of any type of cancer other those listed and their treatment. :download:`.docx <./write-ups/OtherAdultCancer.docx>`

    * **Prostate Cancer**: Cancer of the prostate and its treatment. :download:`.docx <./write-ups/ProstateCancer.docx>`

* Cardio-metabolic Disorders:
    * **Diabetes Type 2, Hypertension, Stroke, Ischemic Heart Disease, Myocardial Infarction** :download:`.docx <./write-ups/CardioMetabolicDisorders.docx>`

* Injuries:
    * **Road Traffic Injuries**: Injuries arising from road traffic incidents and their treatment. :download:`.docx <./write-ups/RoadTrafficInjuries.docx>`

* Other Non-Communicable Conditions
    * **Chronic Lower Back Pain**: Summary representation of chronic lower back pain as part of a set of common non-communicable conditions. :download:`.docx <./write-ups/CardioMetabolicDisorders.docx>`

    * **Chronic Kidney Disease**: Summary representation of chronic kidney disease as part of a set of common non-communicable conditions. :download:`.docx <./write-ups/CardioMetabolicDisorders.docx>`

    * **Depression**: Depression, self-harm and suicide, and the treatment of depression. :download:`.docx <./write-ups/Depression.docx>`

    * **Epilepsy** Epilepsy and its treatment. :download:`.docx <./write-ups/Epilepsy.docx>`

.. toctree::
   :titlesonly:
   :maxdepth: 1

   hsi_events
