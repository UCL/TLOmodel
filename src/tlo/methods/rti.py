"""
A skeleton template for disease methods.

"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ================Put inj randomizer function here for now====================================

def injrandomizer(number):
    # A function that can be called specifying the number of people affected by RTI injuries
    #  and provides outputs for the number of injuries each person experiences from a RTI event, the location of the
    #  injury, the TLO injury categories and the severity of the injuries. The severity of the injuries will then be
    #  used to calculate the injury severity score (ISS), which will then inform mortality and disability

    # Import the distribution of injured body regions from the VIBES study
    totalinjdist = np.genfromtxt('TLOmodel/resources/ResourceFile_RTI_NumberOfInjuredBodyLocations.csv')
    # Import the predicted rate of mortality from the ISS score of injuries
    # ISSmort = np.genfromtxt('AssignInjuryTraits/data/ISSmortality.csv', delimiter=',')
    predinjlocs = []
    predinjsev = []
    predinjcat = []
    predinjiss = []
    predpolytrauma = []
    medintmort = []
    nomedintmort = []
    injlocstring = []
    injcatstring = []
    injaisstring = []





    for n in range(0, number):

        # Reset the distribution of body regions which can injured.
        injlocdist = np.genfromtxt('TLOmodel/resources/ResourceFile_RTI_InjuredBodyRegionPercentage.csv', delimiter=',')

        ninjdecide = np.random.uniform(0, sum(totalinjdist))
        # This generates a random number which will decide how many injuries the person will have,
        # the number of injuries is decided by where the randomly generated number falls on the number line,
        # with regions of the number line designated to the proportions of the nth injury

        cprop = 0
        # cprop is the cumulative frequency of the proportion of total number of injuries, this will be used
        # to find the region of the number line which corresponds to the number of injuries that ninjdecide
        # 'lands' in, which will correspond to the number of injuries

        for i in range(0, len(totalinjdist)):
            # This part of the for loop calculates the cumulative frequency of the proportion of total number of
            # injury, stopping when it finds the region of the cumulative frequency where ninjdecide lies and then
            # uses this to assign a number of injuries, ninj.
            iprop = totalinjdist[i]
            cprop += iprop
            if cprop > ninjdecide:
                ninj = i + 1
                break
        try:
            # This part of the process isn't perfect, but essentially assigns the maximum number of
            # injuries found in the sample if the number of injuries isn't otherwise classified.
            ninj
        except UnboundLocalError:
            ninj = len(totalinjdist)
        # Create an empty vector which will store the injury locations (numberically coded using the
        # abbreviated injury scale coding system, where 1 corresponds to head, 2 to face, 3 to neck, 4 to
        # thorax, 5 to abdomen, 6 to spine, 7 to upper extremity and 8 to lower extremity
        allinjlocs = []
        # Create an empty vector to store the type of injury
        injcat = []
        # Create an empty vector which will store the severity of the injuries
        injais = []
        # print(ninj)

        for j in range(0, ninj):
            upperlim = np.sum(injlocdist[-1:])
            locinvector = np.random.uniform(0, upperlim)
            cat = np.random.uniform(0, 1)
            # loc is a variable which determine which body location will be injured.
            # For each injury assigned to a person, loc will determine the injury location by calculating the
            # cumulative frequency of the proportion of injury location for the jth injury and determining which
            # cumulative frequency boundary loc falls in.
            cprop = 0
            # cumulative proportion
            for k in range(0, len(injlocdist[0])):
                # for the jth injury we find the cumulative frequency of injury location proportion and store it
                # in cprop
                injproprow = injlocdist[1, k]
                cprop += injproprow
                if cprop > locinvector:
                    injlocs = injlocdist[0, k]
                    # Once we find the region of the cumulative frequency of proportion of injury location
                    # loc falls in, we can determine use this to determine where the injury is located, the jth
                    # injury a person has is stored in injlocs initially and then injlocs is stored the vector
                    # allinjlocs and returned as an output at the end of the function
                    allinjlocs.append(int(injlocs))
                    injlocdist = np.delete(injlocdist, k, 1)
                    # print(injlocs)
                    # In injury categories I will use the following mapping:
                    # Fracture - 1
                    # Dislocation - 2
                    # Traumatic brain injury - 3
                    # Soft tissue injury - 4
                    # Internal organ injury - 5
                    # Internal bleeding - 6
                    # Spinal cord injury - 7
                    # Amputation - 8
                    # Eye injury - 9
                    # Cuts etc - 10

                    if injlocs == 1:
                        # stringinjlocs.append('1 ')

                        if cat <= 0.291:
                            injcat.append(int(1))
                            if cat <= 0.91 * 0.291:
                                injais.append(2)
                                # Unspecified skull fracture
                                # print('skull frac')
                            else:
                                injais.append(3)

                                # Basal skull fracture
                                # print('skull frac')

                        elif 0.291 < cat:
                            injcat.append(int(3))
                            # Traumatic brain injuries
                            if cat <= 0.291 + 0.65 * (1 - 0.291):
                                # Mild TBI
                                injais.append(3)

                                # print('mild tbi')

                            elif 0.291 + 0.65 * (1 - 0.291) < cat <= 0.291 + (0.65 + 0.25) * (1 - 0.291):
                                # Moderate TBI
                                injais.append(4)

                                # print('mod tbi')

                            else:
                                # Severe TBI
                                injais.append(5)

                                # print('Sev tbi')
                    if injlocs == 2:
                        # stringinjlocs.append('2 ')
                        if cat <= 0.47:
                            injcat.append(int(1))
                            if cat <= 0.47 * 0.35:
                                # Nasal and unspecified fractures of the face
                                injais.append(1)

                                # print('Face frac 1')
                            else:
                                # Mandible and Zygomatic fractures
                                injais.append(2)

                                # print('Fce frac 1')
                        elif 0.47 < cat <= 0.995:
                            # Skin and soft tissue injury
                            injcat.append(int(10))
                            injais.append(1)

                        else:
                            injcat.append(int(9))
                            injais.append(1)

                            # print('eye inj')
                    if injlocs == 3:
                        # stringinjlocs.append('3 ')
                        if cat <= 0.01:
                            injcat.append(int(4))
                            # Soft tissue injuries of the neck
                            if cat <= 0.005:
                                # Vertebral artery laceration
                                injais.append(2)

                                # print('soft tissue neck 1')
                            else:
                                # Pharynx contusion
                                injais.append(3)

                                # print('soft tissue neck 2')

                        elif 0.01 < cat <= 0.99:
                            # Internal bleeding
                            injcat.append(int(6))
                            if cat < 0.55 * 0.99:
                                # Sternomastoid m. hemorrhage,
                                # Hemorrhage, supraclavicular triangle
                                # Hemorrhage, posterior triangle
                                # Anterior vertebral vessel hemorrhage
                                # Neck muscle hemorrhage
                                injais.append(1)

                                # print('neck internal bleeding 1')
                            else:
                                # Hematoma in carotid sheath
                                # Carotid sheath hemorrhage
                                injais.append(3)

                                # print('neck intern bleed 2')
                        else:
                            # Dislocation
                            injcat.append(int(2))
                            if cat < 0.99666:
                                # Atlanto-axial subluxation
                                injais.append(3)

                                # print('disloc neck 1')
                            else:
                                # Atlanto-occipital subluxation
                                injais.append(2)

                                # print('disloc neck 2')
                    if injlocs == 4:
                        # stringinjlocs.append('4 ')
                        if cat <= 0.4:
                            # Internal Bleeding
                            injcat.append(int(6))
                            if cat <= 0.2:
                                # Chest wall bruises/haematoma
                                injais.append(1)

                                # print('thor intern bled 1')
                            else:
                                # Haemothorax
                                injais.append(3)

                                # print('thor intern bled 2')
                        elif 0.4 < cat <= 0.54:
                            # Internal organ injury
                            injcat.append(int(5))
                            # Lung contusion and Diaphragm rupture
                            injais.append(3)

                            # print('thor internal organ')
                        elif 0.54 < cat <= 0.65:
                            # Fractures to ribs and flail chest
                            injcat.append(int(1))
                            if 0.54 < cat <= 0.54 + 0.8 * 0.11:
                                # fracture to rib(s)
                                injais.append(2)

                            else:
                                # flail chest
                                injais.append(4)

                        else:
                            injcat.append(int(4))
                            if 0.65 < cat <= 0.65 + 0.54 * 0.35:
                                # Chest wall lacerations/avulsions
                                injais.append(1)

                            elif 0.65 + 0.54 * 0.35 < cat <= 0.65 + (0.54 + 0.34) * 0.35:
                                # Open/closed pneumothorax
                                injais.append(3)

                            else:
                                # surgical emphysema
                                injais.append(2)

                    if injlocs == 5:
                        # stringinjlocs.append('5 ')
                        # Internal organ injuries
                        injcat.append(int(5))
                        if cat <= 0.056:
                            # Intestines, Stomach and colon injury
                            injais.append(2)

                            # print('abd int 1')
                        elif 0.056 < cat <= 0.056 + 0.91:
                            # Spleen, bladder, liver, urethra and diaphragm injury
                            injais.append(3)

                            # print('abd int 2')
                        else:
                            # Kidney injury
                            injais.append(4)

                            # print('abd int 3')
                    if injlocs == 6:
                        # stringinjlocs.append('6 ')
                        if cat <= 0.364:
                            # Fracture to vertebrae
                            injcat.append(int(1))
                            injais.append(2)

                            # print('Spine frac')
                        else:
                            # Spinal cord injury
                            injcat.append(int(7))
                            if 0.364 < cat <= 0.364 + 0.09 * (1 - 0.364):
                                injais.append(3)

                                # print('scl1')
                            elif 0.364 + 0.09 * (1 - 0.364) < cat < 0.364 + (0.09 + 0.31) * (1 - 0.364):
                                injais.append(4)

                                # print('scl2')
                            elif 0.364 + (0.09 + 0.31) * (1 - 0.364) < cat < 0.364 + (0.09 + 0.31 + 0.55) * (1 - 0.364):
                                injais.append(5)

                                # print('scl3')
                            else:
                                injais.append(6)

                                # print('scl4')
                    if injlocs == 7:
                        # stringinjlocs.append('7 ')
                        if cat <= 0.943/2:
                            injcat.append(int(10))
                            injais.append(1)
                        elif 0.943/2 < cat <= 0.943:
                            # Fracture to arm
                            injcat.append(int(1))
                            injais.append(2)

                            # print('uxfrac1')
                        elif 0.943 < cat <= 0.943 + 0.002:
                            # Dislocation to arm
                            injcat.append(int(2))
                            injais.append(2)

                            # print('uxdis')
                        elif 0.943 + 0.002 < cat <= 0.943 + 0.002 + 0.051:
                            # Amputation to finger/thumb/unilateral arm
                            injcat.append(int(8))
                            injais.append(2)

                            # print('uxamp1')
                        else:
                            # Amputation, arm, bilateral
                            injcat.append(int(8))
                            injais.append(3)

                            # print('uxamp2')
                    if injlocs == 8:

                        if cat <= 0.006:
                            # Bilateral amputation, leg
                            # print('lx bilat')
                            injcat.append(int(8))
                            injais.append(4)

                        elif 0.006 < cat <= 0.082:
                            # Unilateral amputation, leg
                            injcat.append(int(8))
                            injais.append(3)

                            # print('lx unlat')
                        elif 0.082 < cat <= 0.157:
                            # Amputation of toe
                            injcat.append(int(8))
                            injais.append(2)

                            # print('lx toe')
                        elif 0.157 < cat <= 0.198/2:
                            # Fracture of foot/toes
                            injcat.append(int(1))
                            injais.append(1)

                            # print('lx foot frac')
                        elif 0.198/2 < cat <= 0.813/2:
                            # Fracture of tibia, fibula, patella
                            # print('lx lower leg ')
                            injcat.append(int(1))
                            injais.append(2)

                        elif 0.813/2 < cat <= 0.978/2:
                            # Fracture of femur
                            # print('lx femur')
                            injcat.append(int(1))
                            injais.append(3)

                        elif 0.978/2 < cat <= 0.995:
                            injcat.append(int(10))
                            injais.append(1)

                        elif 0.995 < cat:
                            # Dislocation
                            # print('lx dis')
                            injcat.append(int(2))
                            injais.append(2)

                    # The injury is then assigned an AIS severity ranking based on the possible range of AIS severity
                    # scored for that body region, currently this is random, but could be informed by a distribution
                    break


        # Create a dataframe that stores the injury location and severity for each person, the point of this
        # dataframe is to use some of the pandas tools to manipulate the generated injury data to calculate
        # the ISS score and from this, the probability of mortality resulting from the injuries.
        injlocstring.append(' '.join(map(str, allinjlocs)))
        injcatstring.append(' '.join(map(str, injcat)))
        injaisstring.append(' '.join(map(str, injais)))
        injdata = {'AIS location': allinjlocs, 'AIS severity': injais}
        df = pd.DataFrame(injdata, columns=['AIS location', 'AIS severity'])
        # Find the most severe injury to the person in each body region, creates a new column containing the
        # maximum AIS value of each injured body region
        df['Severity max'] = df.groupby(['AIS location'], sort=False)['AIS severity'].transform(max)
        # column no longer needed and will get in the way of future calculations
        df = df.drop(columns='AIS severity')
        # drops the duplicate values in the location data, preserving the most severe injuries in each body
        # location.
        df = df.drop_duplicates(['AIS location'], keep='first')
        # Finds the AIS score for the most severely injured body regions and stores them in a new dataframe z
        z = df.nlargest(3, 'Severity max', 'first')
        # Find the 3 most severely injured body regions
        z = z.iloc[:3]
        # Need to determine whether the persons injuries qualify as polytrauma as such injuries have a different
        # prognosis, set default as False. Polytrauma is defined via the new Berlin definition, 'when two or more
        # injuries have an AIS severity score of 3 or higher'.
        polytrauma = False
        # Determine where more than one injured body region has occurred
        if len(z) > 1:
            # Find where the injuries have an AIS score of 3 or higher
            cond = np.where(z['Severity max'] > 2)
            if len(z.iloc[cond]) > 1:
                # if two or more injuries have a AIS score of 3 or higher then this person has polytrauma.
                polytrauma = True
        # Calculate the squares of the AIS scores for the three most severely injured body regions
        z['sqrsev'] = z['Severity max'] ** 2
        # From the squared AIS scores, calculate the ISS score
        ISSscore = sum(z['sqrsev'])
        # Use ISS score to determine the percentage of mortality
        # If there is a fatal injury (AIS score > 5) then assume this injury is fatal.

        # ====================Dose Response=======================
        # if ISSscore > 74:
        #     ISSscore = 75
        #     ISSpercmort = 1
        # else:
        #     ISSpercmort = ISSmort[ISSscore]

        # ==================Bounded Mortality====================

        if ISSscore <= 15:
            ISSpercmort = 0.046
        elif 15 < ISSscore <= 74:
            ISSpercmort = 0.27
        elif ISSscore > 74:
            ISSscore = 75
            ISSpercmort = 1

        # Include effects of polytrauma here
        if polytrauma == True:
            pass
            ISSpercmort = 1.9 * ISSpercmort
            if ISSpercmort > 1:
                ISSpercmort = 1

        # Turn the vectors into a string to store as one entry in a dataframe
        allinjlocs = np.array(allinjlocs)
        allinjlocs = allinjlocs.astype(int)
        allinjlocs = ''.join([str(elem) for elem in allinjlocs])
        predinjlocs.append(allinjlocs)
        predinjsev.append(injais)
        predinjcat.append(injcat)
        predinjiss.append(ISSscore)
        predpolytrauma.append(polytrauma)
        medintmort.append(ISSpercmort)
        nomedintmort.append(1 if ISSscore >= 9 else 0.07)
    injdf = pd.DataFrame()
    injdf['Injury locations'] = predinjlocs
    injdf['Injury locations string'] = injlocstring
    injdf['Injury AIS'] = predinjsev
    injdf['Injury AIS string'] = injaisstring
    injdf['Injury category'] = predinjcat
    injdf['Injury category string'] = injcatstring
    injdf['ISS'] = predinjiss
    injdf['Polytrauma'] = predpolytrauma
    injdf['Percent mortality with treatment'] = medintmort
    injdf['Percent mortality without treatment'] = nomedintmort
    injurycategories = injdf['Injury category string'].str.split(expand=True)
    injurylocations = injdf['Injury locations string'].str.split(expand=True)
    injuryais = injdf['Injury AIS string'].str.split(expand=True)
    injurydescription = injurylocations + injurycategories + injuryais
    for (columnname, columndata) in injurydescription.iteritems():
        injurydescription.rename(columns={injurydescription.columns[columnname]: "Injury " + str(columnname + 1)},
                                 inplace=True)

    injurydescription = injurydescription.fillna('na')
    return injdf, injurydescription


class RTI(Module):
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    """
    RTI module for the TLO model
    """

    # Module parameters
    PARAMETERS = {
        'base_rate_injrti': Parameter(
            Types.REAL,
            'Base rate of RTI per year',
        ),
        'rr_injrti_age1829': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 18-29 compared to base rate of RTI',
        ),
        'rr_injrti_age3039': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 30-39 compared to base rate of RTI',
        ),
        'rr_injrti_age4049': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 40-49 compared to base rate of RTI',
        ),
        'rr_injrti_male': Parameter(
            Types.REAL,
            'risk ratio of RTI when male compared to females',
        ),
        'rr_injrti_excessalcohol': Parameter(
            Types.REAL,
            'risk ratio of RTI in those that consume excess alcohol compared to those who do not'
        ),
        'imm_death_proportion_rti': Parameter(
            Types.REAL,
            'Proportion of those involved in an RTI that die at site of accident or die before seeking medical '
            'intervention'
        ),
        'prop_death_post_med_ISS_<=_15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prop_death_post_med_ISS_>15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score of 15 '
        ),
        'prop_death_no_med_ISS_<=_15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month with no treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prop_death_no_med_ISS_>15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month with no treatment for injuries with an ISS'
            'score of 15 '
        ),
        'prob_perm_disability_with_treatment_severe_TBI': Parameter(
            Types.REAL,
            'probability that someone with a treated severe TBI is permanently disabled'
        ),
        'daly_wt_unspecified_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_unspecified_skull_fracture - code 1674'
        ),
        'daly_wt_basilar_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_basilar_skull_fracture - code 1675'
        ),
        'daly_wt_epidural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_epidural_hematoma - code 1676'
        ),
        'daly_wt_subdural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subdural_hematoma - code 1677'
        ),
        'daly_wt_subarachnoid_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subarachnoid_hematoma - code 1678'
        ),
        'daly_wt_brain_contusion': Parameter(
            Types.REAL,
            'daly_wt_brain_contusion - code 1679'
        ),
        'daly_wt_intraventricular_haemorrhage': Parameter(
            Types.REAL,
            'daly_wt_intraventricular_haemorrhage - code 1680'
        ),
        'daly_wt_diffuse_axonal_injury': Parameter(
            Types.REAL,
            'daly_wt_diffuse_axonal_injury - code 1681'
        ),
        'daly_wt_subgaleal_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subgaleal_hematoma - code 1682'
        ),
        'daly_wt_midline_shift': Parameter(
            Types.REAL,
            'daly_wt_midline_shift - code 1683'
        ),
        'daly_wt_facial_fracture': Parameter(
            Types.REAL,
            'daly_wt_facial_fracture - code 1684'
        ),
        'daly_wt_facial_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_facial_soft_tissue_injury - code 1685'
        ),
        'daly_wt_eye_injury': Parameter(
            Types.REAL,
            'daly_wt_eye_injury - code 1686'
        ),
        'daly_wt_neck_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_neck_soft_tissue_injury - code 1687'
        ),
        'daly_wt_neck_internal_bleeding': Parameter(
            Types.REAL,
            'daly_wt_neck_internal_bleeding - code 1688'
        ),
        'daly_wt_neck_dislocation': Parameter(
            Types.REAL,
            'daly_wt_neck_dislocation - code 1689'
        ),
        'daly_wt_chest_wall_bruises_hematoma': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_bruises_hematoma - code 1690'
        ),
        'daly_wt_hemothorax': Parameter(
            Types.REAL,
            'daly_wt_hemothorax - code 1691'
        ),
        'daly_wt_lung_contusion': Parameter(
            Types.REAL,
            'daly_wt_lung_contusion - code 1692'
        ),
        'daly_wt_diaphragm_rupture': Parameter(
            Types.REAL,
            'daly_wt_diaphragm_rupture - code 1693'
        ),
        'daly_wt_rib_fracture': Parameter(
            Types.REAL,
            'daly_wt_rib_fracture - code 1694'
        ),
        'daly_wt_flail_chest': Parameter(
            Types.REAL,
            'daly_wt_flail_chest - code 1695'
        ),
        'daly_wt_chest_wall_laceration': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_laceration - code 1696'
        ),
        'daly_wt_closed_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_closed_pneumothorax - code 1697'
        ),
        'daly_wt_open_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_open_pneumothorax - code 1698'
        ),
        'daly_wt_surgical_emphysema': Parameter(
            Types.REAL,
            'daly_wt_surgical_emphysema aka subcuteal emphysema - code 1699'
        ),
        'daly_wt_abd_internal_organ_injury': Parameter(
            Types.REAL,
            'daly_wt_abd_internal_organ_injury - code 1700'
        ),
        'daly_wt_spinal_cord_lesion_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_with_treatment - code 1701'
        ),
        'daly_wt_spinal_cord_lesion_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_without_treatment - code 1702'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment - code 1703'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment - code 1704'
        ),
        'daly_wt_vertebrae_fracture': Parameter(
            Types.REAL,
            'daly_wt_vertebrae_fracture - code 1705'
        ),
        'daly_wt_clavicle_scapula_humerus_fracture': Parameter(
            Types.REAL,
            'daly_wt_clavicle_scapula_humerus_fracture - code 1706'
        ),
        'daly_wt_hand_wrist_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_with_treatment - code 1707'
        ),
        'daly_wt_hand_wrist_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_without_treatment - code 1708'
        ),
        'daly_wt_radius_ulna_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment - code 1709'
        ),
        'daly_wt_radius_ulna_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_long_term_without_treatment - code 1710'
        ),
        'daly_wt_dislocated_shoulder': Parameter(
            Types.REAL,
            'daly_wt_dislocated_shoulder - code 1711'
        ),
        'daly_wt_amputated_finger': Parameter(
            Types.REAL,
            'daly_wt_amputated_finger - code 1712'
        ),
        'daly_wt_amputated_thumb': Parameter(
            Types.REAL,
            'daly_wt_amputated_thumb - code 1713'
        ),
        'daly_wt_unilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_with_treatment - code 1714'
        ),
        'daly_wt_unilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_without_treatment - code 1715'
        ),
        'daly_wt_bilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_with_treatment - code 1716'
        ),
        'daly_wt_bilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_without_treatment - code 1717'
        ),
        'daly_wt_foot_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_short_term_with_without_treatment - code 1718'
        ),
        'daly_wt_foot_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_long_term_without_treatment - code 1719'
        ),
        'daly_wt_patella_tibia_fibula_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_with_treatment - code 1720'
        ),
        'daly_wt_patella_tibia_fibula_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_without_treatment - code 1721'
        ),
        'daly_wt_hip_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_short_term_with_without_treatment - code 1722'
        ),
        'daly_wt_hip_fracture_long_term_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_with_treatment - code 1723'
        ),
        'daly_wt_hip_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_without_treatment - code 1724'
        ),
        'daly_wt_pelvis_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_short_term - code 1725'
        ),
        'daly_wt_pelvis_fracture_long_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_long_term - code 1726'
        ),
        'daly_wt_femur_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_short_term - code 1727'
        ),
        'daly_wt_femur_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_long_term_without_treatment - code 1728'
        ),
        'daly_wt_dislocated_hip': Parameter(
            Types.REAL,
            'daly_wt_dislocated_hip - code 1729'
        ),
        'daly_wt_dislocated_knee': Parameter(
            Types.REAL,
            'daly_wt_dislocated_knee - code 1730'
        ),
        'daly_wt_amputated_toes': Parameter(
            Types.REAL,
            'daly_wt_amputated_toes - code 1731'
        ),
        'daly_wt_unilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_with_treatment - code 1732'
        ),
        'daly_wt_unilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_without_treatment - code 1733'
        ),
        'daly_wt_bilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_with_treatment - code 1734'
        ),
        'daly_wt_bilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_without_treatment - code 1735'
        ),
    }

    PROPERTIES = {
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
        'rt_injseverity': Property(Types.CATEGORICAL,
                                   'Injury status relating to road traffic injury: none, mild, moderate, severe',
                                   categories=['none', 'mild', 'severe'],
                                   ),
        'rt_fracture': Property(Types.CATEGORICAL, 'fractured bone resulting from RTI',
                                categories=['none', 'foot', 'hip', 'patella/tibia/fibula/ankle', 'pelvis',
                                            'femur other than femoral neck', 'clavicle, scapula, humerus', 'hand/wrist',
                                            'radius/ulna', 'vertebrae', 'fractured ribs', 'flail chest', 'mandible',
                                            'nasal', 'zygomatic', 'unspfacial', 'basilar skull', 'unspskull'
                                            ]),
        'rt_fracture_diagnosed': Property(Types.BOOL, 'fractured bone diagnosed'),
        'rt_fracture_treated': Property(Types.BOOL, 'fractured bone treated'),
        'rt_dislocation': Property(Types.CATEGORICAL, 'dislocated joint resulting from RTI',
                                   categories=['none', 'atlanto-axial subluxation', 'atlanto-occipital subluxation',
                                               'hip', 'knee', 'shoulder']),
        'rt_dislocation_diagnosed': Property(Types.BOOL, 'dislocated joint diagnosed'),
        'rt_dislocation_treated': Property(Types.BOOL, 'dislocated joint treated'),
        'rt_tbi': Property(Types.CATEGORICAL, 'traumatic brain injury resulting from RTI, mild, moderate, severe',
                           categories=['none', 'epidural hematoma', 'subdural hemtaoma', 'subarachnoid haemorrhage',
                                       'contusion', 'intraventricular haemorrhage', 'diffuse axonal injury',
                                       'subgaleal hematoma', 'midline shift']),
        'rt_tbi_diagnosed': Property(Types.BOOL, 'traumatic brain injury diagnosed'),
        'rt_tbi_treated': Property(Types.BOOL, 'traumatic brain injury treated'),
        'rt_tbi_recovered': Property(Types.BOOL, 'recovery from traumatic brain injury'),
        'rt_soft': Property(Types.CATEGORICAL, 'soft tissue injury resulting from RTI.',
                            categories=['none', 'unspface', 'vertebral artery laceration', 'pharynx contusion',
                                        'chest wall lacerations/avulsions', 'closed pneumothorax',
                                        'open pneumothorax', 'subcutaneous emphysema']),
        'rt_soft_diagnosed': Property(Types.BOOL, 'soft tissue injury resulting from RTI diagnosed'),
        'rt_soft_treated': Property(Types.BOOL, 'soft tissue injury treated'),
        'rt_ioi': Property(Types.CATEGORICAL, 'internal organ injury resulting from RTI.',
                           categories=['none', 'lung contusion', 'diaphragm rupture', 'spleen', 'urinary bladder',
                                       'intestines', 'liver', 'urethra', 'stomach', 'colon', 'kidney']),
        'rt_ioi_diagnosed': Property(Types.BOOL, 'internal organ injury diagnosed'),
        'rt_ioi_treated': Property(Types.BOOL, 'internal organ injury treated'),
        'rt_intbleed': Property(Types.CATEGORICAL, 'internal bleeding resulting from RTI.',
                                categories=['none', 'sternomastoid m. hemorrhage',
                                            'supraclavicular triangle Hemorrhage', 'posterior triangle hemorrhage',
                                            'anterior vertebral vessel hemorrhage', 'hematoma in carotid sheath',
                                            'carotid sheath hemorrhage', 'neck muscle hemorrhage',
                                            'chest wall bruises/haematoma', 'haemothorax']),
        'rt_intbleed_diagnosed': Property(Types.BOOL, 'internal bleeding diagnosed'),
        'rt_intbleed_treated': Property(Types.BOOL, 'internal bleeding treated'),
        'rt_sci': Property(Types.CATEGORICAL, 'spinal cord injury from RTI at neck/below neck level',
                           categories=['none', 'neck', 'below neck']),
        'rt_sci_diagnosed': Property(Types.BOOL, 'spinal cord injury from RTI diagnosed'),
        'rt_sci_treated': Property(Types.BOOL, 'spinal cord injury from RTI treated'),
        'rt_sci_recovered': Property(Types.BOOL, 'recovery from spinal cord injury'),
        'rt_amp': Property(Types.CATEGORICAL, 'amputation from RTI.',
                           categories=['none', 'thumb', 'finger', 'unilateral upper limb', 'bilateral upper limb',
                                       'toe', 'unilateral lower limb', 'bilateral lower limb']),
        'rt_amp_diagnosed': Property(Types.BOOL, 'amputation from RTI diagnosed'),
        'rt_amp_treated': Property(Types.BOOL, 'amputation from RTI treated'),
        'rt_wound': Property(Types.BOOL, 'wound from rti.'),
        'rt_wound_diagnosed': Property(Types.BOOL, 'wound diagnosed'),
        'rt_wound_treated': Property(Types.BOOL, 'wound treated'),
        'rt_eye_inj': Property(Types.BOOL, 'eye injury from RTI'),
        'rt_eye_inj_diagnosed': Property(Types.BOOL, 'eye injury from RTI diagnosed'),
        'rt_eye_inj_treated': Property(Types.BOOL, 'eye injury from RTI treated'),
        'rt_perm_disability': Property(Types.BOOL, 'whether the injuries from an RTI result in permanent disability'),
        'rt_polytrauma': Property(Types.BOOL, 'polytrauma from RTI'),
        'rt_imm_death': Property(Types.BOOL, 'death at scene True/False'),
        'rt_med_int': Property(Types.BOOL, 'medical intervention True/False'),
        'rt_recovery_no_med': Property(Types.BOOL, 'recovery without medical intervention True/False'),
        'rt_post_med_death': Property(Types.BOOL, 'death in following month despite medical intervention True/False'),
        'rt_disability': Property(Types.REAL, 'disability weight for current month'),
        'rt_date_inj': Property(Types.DATE, 'date of latest injury')
    }

    # generic symptom for severely traumatic injuries, mild injuries accounted for in generic symptoms under 'injury'

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        if "HealthBurden" in self.sim.modules.keys():
            # get the DALY weights of the seq associated with road traffic injuries
            self.parameters["daly_wt_unspecified_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1674
            )
            self.parameters["daly_wt_basilar_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1675
            )
            self.parameters["daly_wt_epidural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1676
            )
            self.parameters["daly_wt_subdural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1677
            )
            self.parameters["daly_wt_subarachnoid_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1678
            )
            self.parameters["daly_wt_brain_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1679
            )
            self.parameters["daly_wt_intraventricular_haemorrhage"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1680
            )
            self.parameters["daly_wt_diffuse_axonal_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1681
            )
            self.parameters["daly_wt_subgaleal_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1682
            )
            self.parameters["daly_wt_midline_shift"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1683
            )
            self.parameters["daly_wt_facial_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1684
            )
            self.parameters["daly_wt_facial_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1685
            )
            self.parameters["daly_wt_eye_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1686
            )
            self.parameters["daly_wt_neck_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1687
            )
            self.parameters["daly_wt_neck_internal_bleeding"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1688
            )
            self.parameters["daly_wt_neck_dislocation"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1689
            )
            self.parameters["daly_wt_chest_wall_bruises_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1690
            )
            self.parameters["daly_wt_hemothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1691
            )
            self.parameters["daly_wt_lung_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1692
            )
            self.parameters["daly_wt_diaphragm_rupture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1693
            )
            self.parameters["daly_wt_rib_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1694
            )
            self.parameters["daly_wt_flail_chest"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1695
            )
            self.parameters["daly_wt_chest_wall_laceration"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1696
            )
            self.parameters["daly_wt_closed_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1697
            )
            self.parameters["daly_wt_open_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1698
            )
            self.parameters["daly_wt_surgical_emphysema"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1699
            )
            self.parameters["daly_wt_abd_internal_organ_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1700
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1701
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1702
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1703
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1704
            )
            self.parameters["daly_wt_vertebrae_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1705
            )
            self.parameters["daly_wt_clavicle_scapula_humerus_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1706
            )
            self.parameters["daly_wt_hand_wrist_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1707
            )
            self.parameters["daly_wt_hand_wrist_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1708
            )
            self.parameters["daly_wt_radius_ulna_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1709
            )
            self.parameters["daly_wt_radius_ulna_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1710
            )
            self.parameters["daly_wt_dislocated_shoulder"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1711
            )
            self.parameters["daly_wt_amputated_finger"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1712
            )
            self.parameters["daly_wt_amputated_thumb"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1713
            )
            self.parameters["daly_wt_unilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1714
            )
            self.parameters["daly_wt_unilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1715
            )
            self.parameters["daly_wt_bilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1716
            )
            self.parameters["daly_wt_bilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1717
            )
            self.parameters["daly_wt_foot_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1718
            )
            self.parameters["daly_wt_foot_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1719
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1720
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1721
            )
            self.parameters["daly_wt_hip_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1722
            )
            self.parameters["daly_wt_hip_fracture_long_term_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1723
            )
            self.parameters["daly_wt_hip_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1724
            )
            self.parameters["daly_wt_pelvis_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1725
            )
            self.parameters["daly_wt_pelvis_fracture_long_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1726
            )
            self.parameters["daly_wt_femur_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1727
            )
            self.parameters["daly_wt_femur_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1728
            )
            self.parameters["daly_wt_dislocated_hip"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1729
            )
            self.parameters["daly_wt_dislocated_knee"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1730
            )
            self.parameters["daly_wt_amputated_toes"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1731
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1732
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1733
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1734
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1735
            )

    # Declare the non-generic symptoms that this module will use.
    # It will not be able to use any that are not declared here. They do not need to be unique to this module.
    # You should not declare symptoms that are generic here (i.e. in the generic list of symptoms)

    SYMPTOMS = {'em_severe_trauma',  # Generic for severe injuries.
                # Fracture
                'bleeding from wound',
                'bruising around trauma site',
                'severe pain at trauma site',
                'swelling around trauma site',
                'redness or warmth around trauma site',
                'visual disturbances',
                'restlessness',
                'irritability',
                'loss of balance',
                'stiffness',
                'abnormal pupil behaviour/reflexes',
                'confusion',
                'fatigue',
                'fainting',
                'excessive salivation',
                'difficulty swallowing',
                'nosebleed',
                'breathing difficulty',
                'audible signs of injury',
                'uneven chest rise',
                'seat belt marks',
                'visual deformity of body part',
                'limitation of movement',
                'inability to walk',
                # TBI
                'periorbital ecchymosis',
                'shock',
                'hyperbilirubinemia',
                'abnormal posturing',
                'nausea',
                'loss of consciousness',
                'coma',
                'seizures',
                'tinnitus',
                'sensitive to light',
                'slurred speech',
                'personality change',
                'paralysis',
                'weakness in one half of body',
                # Dislocation
                'numbness in lower back and lower limbs',
                'muscle spasms',
                'hypermobile patella'
                # Soft tissue injury
                'ataxia',
                'coughing up blood',
                'stridor',
                'subcutaneous air',
                'blue discoloration of skin or lips',
                'pressure in chest',
                'rapid breathing',
                # Internal organ injury
                'low blood pressure',
                'Bluish discoloration of the belly',
                'Right-sided abdominal pain and right shoulder pain',
                'Blood in the urine',
                'Left arm and shoulder pain',
                'rigid abdomen',
                'cyanosis',
                'heart palpitations',
                'pain in the left shoulder or left side of the chest',
                'difficulty urinating',
                'urine leakage',
                'abdominal distension',
                'rectal bleeding',
                # Internal bleeding
                'sweaty skin',
                # Spinal cord injury
                'inability to control bladder',
                'inability to control bowel',
                'unnatural positioning of the head',
                # Amputation - limb's bloody gone
                }

    def initialise_population(self, population):
        df = population.props
        now = self.sim.date.year
        df.loc[df.is_alive & 'rt_road_traffic_inc'] = False
        df.loc[df.is_alive & 'rt_injseverity'] = "none"  # default: no one has been injured in a RTI
        df.loc[df.is_alive & 'rt_fracture'] = "none"  # default: no fractures at birth
        df.loc[df.is_alive & 'rt_fracture_diagnosed'] = False
        df.loc[df.is_alive & 'rt_fracture_treated'] = False
        df.loc[df.is_alive & 'rt_dislocation'] = "none"  # default: no dislocations at birth
        df.loc[df.is_alive & 'rt_dislocation_diagnosed'] = False
        df.loc[df.is_alive & 'rt_dislocation_treated'] = False
        df.loc[df.is_alive & 'rt_tbi'] = "none"  # default: no traumatic brain injury at birth
        df.loc[df.is_alive & 'rt_tbi_diagnosed'] = False
        df.loc[df.is_alive & 'rt_rbi_treated'] = False
        df.loc[df.is_alive & 'rt_tbi_recovered'] = False
        df.loc[df.is_alive & 'rt_soft'] = "none"  # default: no soft tissue injury at birth
        df.loc[df.is_alive & 'rt_soft_diagnosed'] = False
        df.loc[df.is_alive & 'rt_soft_treated'] = False
        df.loc[df.is_alive & 'rt_ioi'] = "none"  # default: no internal organ injury at birth
        df.loc[df.is_alive & 'rt_ioi_diagnosed'] = False
        df.loc[df.is_alive & 'rt_ioi_treated'] = False
        df.loc[df.is_alive & 'rt_intbleed'] = "none"  # default: no internal bleeding at birth
        df.loc[df.is_alive & 'rt_intbleed_diagnosed'] = False
        df.loc[df.is_alive & 'rt_intbleed_treated'] = False
        df.loc[df.is_alive & 'rt_sci'] = "none"
        df.loc[df.is_alive & 'rt_sci_diagnosed'] = False
        df.loc[df.is_alive & 'rt_sci_treated'] = False
        df.loc[df.is_alive & 'rt_amp'] = "none"
        df.loc[df.is_alive & 'rt_amp_diagnosed'] = False
        df.loc[df.is_alive & 'rt_amp_treated'] = False
        df.loc[df.is_alive & 'rt_eye_inj'] = False
        df.loc[df.is_alive & 'rt_eye_inj_diagnosed'] = False
        df.loc[df.is_alive & 'rt_eye_inj_treated'] = False
        df.loc[df.is_alive & 'rt_wound'] = False
        df.loc[df.is_alive & 'rt_wound_diagnosed'] = False
        df.loc[df.is_alive & 'rt_wound_treated'] = False
        df.loc[df.is_alive & 'rt_polytrauma'] = False
        df.loc[df.is_alive & 'rt_perm_disability'] = False
        df.loc[df.is_alive & 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.loc[df.is_alive & 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.loc[df.is_alive & 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.loc[df.is_alive & 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.loc[df.is_alive & 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive & 'rt_date_inj'] = pd.NaT

    def initialise_simulation(self, sim):
        """Add lifestyle events to this simulation

        """
        event = RTIEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        event = RTILoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Register this disease module with the health system
        self.sim.modules["HealthSystem"].register_disease_module(self)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.loc[df.is_alive & 'rt_road_traffic_inc'] = False
        df.at[child_id, 'rt_injseverity'] = "none"  # default: no one has been injured in a RTI
        df.at[child_id, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.at[child_id, 'rt_fracture'] = "none"  # default: no fractures at birth
        df.at[child_id, 'rt_fracture_diagnosed'] = False
        df.at[child_id, 'rt_fracture_treated'] = False
        df.at[child_id, 'rt_dislocation'] = "none"  # default: no dislocations at birth
        df.at[child_id, 'rt_dislocation_diagnosed'] = False
        df.at[child_id, 'rt_dislocation_treated'] = False
        df.at[child_id, 'rt_tbi'] = "none"  # default: no traumatic brain injury at birth
        df.at[child_id, 'rt_tbi_diagnosed'] = False
        df.at[child_id, 'rt_tbi_treated'] = False
        df.at[child_id, 'rt_soft'] = "none"  # default: no soft tissue injury at birth
        df.at[child_id, 'rt_soft_diagnosed'] = False
        df.at[child_id, 'rt_soft_treated'] = False
        df.at[child_id, 'rt_ioi'] = "none"  # default: no internal organ injury at birth
        df.at[child_id, 'rt_ioi_diagnosed'] = False
        df.at[child_id, 'rt_ioi_treated'] = False
        df.at[child_id, 'rt_intbleed'] = "none"  # default: no internal bleeding at birth
        df.at[child_id, 'rt_intbleed_diagnosed'] = False
        df.at[child_id, 'rt_intbleed_treated'] = False
        df.at[child_id, 'rt_sci'] = "none"
        df.at[child_id, 'rt_sci_diagnosed'] = False
        df.at[child_id, 'rt_sci_treated'] = False
        df.at[child_id, 'rt_amp'] = "none"
        df.at[child_id, 'rt_amp_diagnosed'] = False
        df.at[child_id, 'rt_amp_treated'] = False
        df.at[child_id, 'rt_eye_inj'] = False
        df.at[child_id, 'rt_eye_inj_diagnosed'] = False
        df.at[child_id, 'rt_eye_inj_treated'] = False
        df.at[child_id, 'rt_wound'] = False
        df.at[child_id, 'rt_wound_diagnosed'] = False
        df.at[child_id, 'rt_wound_treated'] = False
        df.at[child_id, 'rt_polytrauma'] = False
        df.at[child_id, 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.at[child_id, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.at[child_id, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.at[child_id, 'rt_disability'] = 0  # default: no disability due to RTI
        df.at[child_id, 'rt_date_inj'] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(
            'This is RTI, being alerted about a health system interaction person %d for: %s',
            person_id,
            treatment_id,
        )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug('This is RTI reporting my daly values')
        df = self.sim.population.props
        disability_series_for_alive_persons = df.loc[df["is_alive"], "rt_disability"]
        return disability_series_for_alive_persons


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class RTIEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """
    # todo: check list of parameters over
    def __init__(self, module):
        """Shedule to take place every month
        """
        super().__init__(module, frequency=DateOffset(months=1))
        p = module.parameters
        self.base_1m_prob_rti = p['base_rate_injrti'] / 12
        self.rr_injrti_age1829 = p['rr_injrti_age1829']
        self.rr_injrti_age3039 = p['rr_injrti_age3039']
        self.rr_injrti_age4049 = p['rr_injrti_age4049']
        self.rr_injrti_male = p['rr_injrti_male']
        self.rr_injrti_excessalcohol = p['rr_injrti_excessalcohol']
        self.imm_death_proportion_rti = p['imm_death_proportion_rti']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.daly_wt_unspecified_skull_fracture = p['daly_wt_unspecified_skull_fracture']
        self.daly_wt_basilar_skull_fracture = p['daly_wt_basilar_skull_fracture']
        self.daly_wt_epidural_hematoma = p['daly_wt_epidural_hematoma']
        self.daly_wt_subdural_hematoma = p['daly_wt_subdural_hematoma']
        self.daly_wt_subarachnoid_hematoma = p['daly_wt_subarachnoid_hematoma']
        self.daly_wt_brain_contusion = p['daly_wt_brain_contusion']
        self.daly_wt_intraventricular_haemorrhage = p['daly_wt_intraventricular_haemorrhage']
        self.daly_wt_diffuse_axonal_injury = p['daly_wt_diffuse_axonal_injury']
        self.daly_wt_subgaleal_hematoma = p['daly_wt_subgaleal_hematoma']
        self.daly_wt_midline_shift = p['daly_wt_midline_shift']
        self.daly_wt_facial_fracture = p['daly_wt_facial_fracture']
        self.daly_wt_facial_soft_tissue_injury = p['daly_wt_facial_soft_tissue_injury']
        self.daly_wt_eye_injury = p['daly_wt_eye_injury']
        self.daly_wt_neck_soft_tissue_injury = p['daly_wt_neck_soft_tissue_injury']
        self.daly_wt_neck_internal_bleeding = p['daly_wt_neck_internal_bleeding']
        self.daly_wt_neck_dislocation = p['daly_wt_neck_dislocation']
        self.daly_wt_chest_wall_bruises_hematoma = p['daly_wt_chest_wall_bruises_hematoma']
        self.daly_wt_hemothorax = p['daly_wt_hemothorax']
        self.daly_wt_lung_contusion = p['daly_wt_lung_contusion']
        self.daly_wt_diaphragm_rupture = p['daly_wt_diaphragm_rupture']
        self.daly_wt_rib_fracture = p['daly_wt_rib_fracture']
        self.daly_wt_flail_chest = p['daly_wt_flail_chest']
        self.daly_wt_chest_wall_laceration = p['daly_wt_chest_wall_laceration']
        self.daly_wt_closed_pneumothorax = p['daly_wt_closed_pneumothorax']
        self.daly_wt_open_pneumothorax = p['daly_wt_open_pneumothorax']
        self.daly_wt_surgical_emphysema = p['daly_wt_surgical_emphysema']
        self.daly_wt_abd_internal_organ_injury = p['daly_wt_abd_internal_organ_injury']
        self.daly_wt_spinal_cord_lesion_neck_with_treatment = p['daly_wt_spinal_cord_lesion_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_neck_without_treatment = p['daly_wt_spinal_cord_lesion_neck_without_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_with_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_without_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment']
        self.daly_wt_vertebrae_fracture = p['daly_wt_vertebrae_fracture']
        self.daly_wt_clavicle_scapula_humerus_fracture = p['daly_wt_clavicle_scapula_humerus_fracture']
        self.daly_wt_hand_wrist_fracture_with_treatment = p['daly_wt_hand_wrist_fracture_with_treatment']
        self.daly_wt_hand_wrist_fracture_without_treatment = p['daly_wt_hand_wrist_fracture_without_treatment']
        self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment = p[
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment']
        self.daly_wt_radius_ulna_fracture_long_term_without_treatment = p[
            'daly_wt_radius_ulna_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_shoulder = p['daly_wt_dislocated_shoulder']
        self.daly_wt_amputated_finger = p['daly_wt_amputated_finger']
        self.daly_wt_amputated_thumb = p['daly_wt_amputated_thumb']
        self.daly_wt_unilateral_arm_amputation_with_treatment = p['daly_wt_unilateral_arm_amputation_with_treatment']
        self.daly_wt_unilateral_arm_amputation_without_treatment = p[
            'daly_wt_unilateral_arm_amputation_without_treatment']
        self.daly_wt_bilateral_arm_amputation_with_treatment = p['daly_wt_bilateral_arm_amputation_with_treatment']
        self.daly_wt_bilateral_arm_amputation_without_treatment = p[
            'daly_wt_bilateral_arm_amputation_without_treatment']
        self.daly_wt_foot_fracture_short_term_with_without_treatment = p[
            'daly_wt_foot_fracture_short_term_with_without_treatment']
        self.daly_wt_foot_fracture_long_term_without_treatment = p['daly_wt_foot_fracture_long_term_without_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_with_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_with_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_without_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_without_treatment']
        self.daly_wt_hip_fracture_short_term_with_without_treatment = p[
            'daly_wt_hip_fracture_short_term_with_without_treatment']
        self.daly_wt_hip_fracture_long_term_with_treatment = p['daly_wt_hip_fracture_long_term_with_treatment']
        self.daly_wt_hip_fracture_long_term_without_treatment = p['daly_wt_hip_fracture_long_term_without_treatment']
        self.daly_wt_pelvis_fracture_short_term = p['daly_wt_pelvis_fracture_short_term']
        self.daly_wt_pelvis_fracture_long_term = p['daly_wt_pelvis_fracture_long_term']
        self.daly_wt_femur_fracture_short_term = p['daly_wt_femur_fracture_short_term']
        self.daly_wt_femur_fracture_long_term_without_treatment = p[
            'daly_wt_femur_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_hip = p['daly_wt_dislocated_hip']
        self.daly_wt_dislocated_knee = p['daly_wt_dislocated_knee']
        self.daly_wt_amputated_toes = p['daly_wt_amputated_toes']
        self.daly_wt_unilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_unilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_without_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_without_treatment']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng
        # set rt_imm_death back to false after death
        df.loc[df.is_alive, "rt_imm_death"] = False
        # set rt_post_med_death back to false after death
        df.loc[df.is_alive, "rt_post_med_death"] = False
        # set rt_disability back to 0 after death
        df.loc[df.is_alive, "rt_disability"] = 0

        # ----------- UPDATING OF RTI OVER TIME ----------------
        rt_current_non_ind = df.index[df.is_alive & (df.rt_road_traffic_inc is False)]
        rt_current_mild_inj_ind = df.index[df.is_alive & (df.rt_roadtrafficinj is "mild") & (df.rt_imm_death is False)]
        rt_current_sev_inj_ind = df.index[df.is_alive & (df.rt_roadtrafficinj is "severe") & (df.rt_imm_death is False)]
        rt_current_imm_dead = df.index[df.is_alive & (df.rt_roadtrafficinj is not "none") & (df.rt_imm_death is True)]
        rt_current_med_int = df.index[df.is_alive & (df.rt_roadtrafficinj is not "none") & (df.rt_med_int is True)]
        rt_current_rec_no_med = df.index[df.is_alive & (df.rt_roadtrafficinj is not "none") &
                                         (df.rt_recovery_no_med is True)]
        # Update for people currently not involved in a RTI, make some involved in a RTI event

        eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                         self.base_1m_prob_rti,
                         Predictor('sex').when('M', self.rr_injrti_male),
                         Predictor('age').when('.between(18, 29)', self.rr_injrti_age1829),
                         Predictor('age').when('.between(30, 39)', self.rr_injrti_age3039),
                         Predictor('age').when('.between(40, 49)', self.rr_injrti_age4049),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(rt_current_non_ind)
        random_draw_in_rti = self.module.rng.random_sample(size=len(rt_current_non_ind))
        selected_for_rti = rt_current_non_ind[pred > random_draw_in_rti]
        if selected_for_rti.sum():
            # Take those involved in a RTI and assign some to death
            in_rti_idx = rt_current_non_ind[selected_for_rti]
            df.loc[in_rti_idx, 'rt_road_traffic_inc'] = True
            random_draw_imm_death = self.module.rng.random_sample(size=len(selected_for_rti))
            idx = df.index[df.is_alive &
                           (df.rt_road_traffic_inc is True)]
            selected_to_die = idx[self.imm_mortality_proportion_rti > random_draw_imm_death]
            df.loc[selected_to_die, 'rt_imm_death'] = True
            for individual_id in selected_to_die:
                self.sim.schedule_event(
                    demography.InstantaneousDeath(self.module, individual_id, "Road_Traffic_Incident_imm_death"),
                    self.sim.date
                )
            # Take those remaining people involved in a RTI and assign injuries to them

            selected_for_rti_inj = df.loc[df.isalive & (df.rt_road_traffic_inc is True) & (df.rt_imm_death is False)]
            mortality, description = injrandomizer(len(selected_for_rti_inj))
            selected_for_rti_inj = pd.concat([selected_for_rti_inj, mortality], axis=1)
            selected_for_rti_inj = pd.concat([selected_for_rti_inj, description], axis=1)
            # ============================ Injury severity classification============================================

            # todo:  get the injuries out of the description data frame

            # Find those with mild injuries and update the rt_roadtrafficinj property so they have a mild injury
            mild_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj.isalive & selected_for_rti_inj['ISS'] < 15]
            df.loc[mild_rti_idx, 'rt_injseverity'] = 'mild'

            # Find those with severe injuries and update the rt_roadtrafficinj property so they have a severe injury
            severe_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj['ISS'] >= 15]
            df.loc[severe_rti_idx, 'rt_injseverity'] = 'severe'

            # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
            polytrauma_idx = selected_for_rti_inj.index[selected_for_rti_inj['Polytrauma'] is True]
            df.loc[polytrauma_idx, 'rt_polytrauma'] = True

            # Find those with fractures and update rt_fracture property to reflect the injury
            fracture_idx = selected_for_rti_inj.index[selected_for_rti_inj['Injury category'].str.find('1') is True]
            # frac_loc_idx = fracture_idx
            # df.loc[fracture_idx, 'rt_fracture'] =


            # Find those with dislocations and update rt_dislocation property to reflect the injury
            disloc_idx = selected_for_rti_inj.index[selected_for_rti_inj['Injury category'].str.match('2') is True]
            df.loc[disloc_idx, 'rt_dislocation'] = True

            # Find those with traumatic brain injury and update rt_tbi to match
            tbi_idx = selected_for_rti_inj.index[selected_for_rti_inj['Injury category'].str.match('3') is True]
            df.loc[tbi_idx, 'rt_tbi'] = True

            # Find those with soft tissue injury and update rt_soft to reflect
            soft_idx = selected_for_rti_inj.index[selected_for_rti_inj['Injury category'].str.match('4') is True]
            df.loc[soft_idx, 'rt_soft'] = True

            # Find those with internal organ injury and update rt_ioi to match
            ioi_idx = selected_for_rti_inj.index[selected_for_rti_inj['Injury category'].str.match('5') is True]
            df.loc[ioi_idx, 'rt_ioi'] = True

            # Find those with internal bleeding and update rt_intbleed to match
            int_bleed_idx = selected_for_rti_inj[selected_for_rti_inj['Injury category'].str.match('6') is True]
            df.loc[int_bleed_idx, 'rt_intbleed'] = True

            # Find those with spinal cord injury and update rt_sci to match
            sci_idx = selected_for_rti_inj[selected_for_rti_inj['Injury category'].str.match('7') is True]
            df.loc[sci_idx, 'rt_sci'] = True

            # Find those with amputations and update rt_amp to match
            amp_idx = selected_for_rti_inj[selected_for_rti_inj['Injury category'].str.match('8') is True]
            df.loc[amp_idx, 'rt_amp'] = True

            # Find those with eye injuries and update rt_eye_inj to match
            eye_idx = selected_for_rti_inj[selected_for_rti_inj['Injury category'].str.match('9') is True]
            df.loc[eye_idx, 'rt_eye_inj'] = True


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class RTILoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, Skeleton)

    def apply(self, population):
        # Make some summary statitics

        dict_to_output = {
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info('%s|summary_12m|%s', self.sim.date, dict_to_output)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_RTI_Traumatic_Brain_Injury(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
    An appointment of a person who has experienced head injury due to a road traffic injury, requiring the resources
    found at a level 1+ facility such as:

    Diagnostic tools -

    (Computed) tomograpy - a.k.a ct scan
    MRI scan
    Diagnostic radiography procedures e.g. x-rays

    Treatments -
    Anti inflammetory drugs
    Major surgery e.g. craniotomy, aspiration

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient
        the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
        the_appt_footprint['MRI'] = 1  # This appointment requires a MRI scan
        the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_TBI_Interaction'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        p = RTI.module.parameters

        df = self.sim.population.props

        df.at[person_id, 'rt_tbi_diagnosed'] = True
        df.at[person_id, 'rt_tbi_treated'] = True
        logger.debug('@@@@@@@@@@ TBI Treatment started !!!!!!')
        prob_dis = np.random.random()

        if prob_dis < p['prob_perm_disability_with_treatment_severe_TBI']:
            df.at[person_id, 'rt_perm_disability'] = True
            logger.debug('@@@@@@@@@@ TBI Treatment started but still disabled!!!!!!')
        else:
            df.at[person_id, 'rt_tbi_recovered'] = True
        pass

    def did_not_run(self):
        logger.debug('HSI_RTI_Traumatic_Brain_Injury: did not run')

        # todo: find out the probability of death without treatment
        #
        #
        #

        pass
