"""
A skeleton template for disease methods.

"""
from pathlib import Path
import pandas as pd
import numpy as np
from tlo import DateOffset, Module, Parameter, Property, Types, logging, Date
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
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
    totalinjdist = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/resources/'
                                 'ResourceFile_RTI_NumberOfInjuredBodyLocations.csv')
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
        injlocdist = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/resources/'
                                   'ResourceFile_RTI_InjuredBodyRegionPercentage.csv', delimiter=',')

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
                        if cat <= 0.943 / 2:
                            injcat.append(int(10))
                            injais.append(1)
                        elif 0.943 / 2 < cat <= 0.943:
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
                        elif 0.157 < cat <= 0.198 / 2:
                            # Fracture of foot/toes
                            injcat.append(int(1))
                            injais.append(1)

                            # print('lx foot frac')
                        elif 0.198 / 2 < cat <= 0.813 / 2:
                            # Fracture of tibia, fibula, patella
                            # print('lx lower leg ')
                            injcat.append(int(1))
                            injais.append(2)

                        elif 0.813 / 2 < cat <= 0.978 / 2:
                            # Fracture of femur
                            # print('lx femur')
                            injcat.append(int(1))
                            injais.append(3)

                        elif 0.978 / 2 < cat <= 0.995:
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
        if polytrauma is True:
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
    injdf['Injury category string'] = injdf['Injury category string'].astype(str)
    injurycategories = injdf['Injury category string'].str.split(expand=True)
    injdf['Injury locations string'] = injdf['Injury locations string'].astype(str)
    injurylocations = injdf['Injury locations string'].str.split(expand=True)
    injdf['Injury AIS string'] = injdf['Injury AIS string'].astype(str)
    injuryais = injdf['Injury AIS string'].str.split(expand=True)
    injurydescription = injurylocations + injurycategories + injuryais
    injurydescription = injurydescription.astype(str)
    for (columnname, columndata) in injurydescription.iteritems():
        injurydescription.rename(columns={injurydescription.columns[columnname]: "Injury " + str(columnname + 1)},
                                 inplace=True)

    injurydescription = injurydescription.fillna("none")
    return injdf, injurydescription


# def find_and_count_injuries(dataframe, tloinjcodes):
#     # This function searches the dataframe and finds who has been affected by an injury
#     index = pd.Index([])
#     counts = 0
#     for code in tloinjcodes:
#         inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(), axis=1)
#         injidx = inj.index[inj]
#         counts += len(injidx)
#         index = index.union(injidx)
#     return index, counts

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
        'prob_death_with_med_mild': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prob_death_with_med_severe': Parameter(
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
        'prob_perm_disability_with_treatment_sci': Parameter(
            Types.REAL,
            'probability that someone with a treated spinal cord injury is permanently disabled'
        ),
        'prob_death_TBI_SCI_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a spinal cord injury will die without treatment'
        ),
        'prob_TBI_require_craniotomy': Parameter(
            Types.REAL,
            'probability that someone with a traumatic brain injury will require a craniotomy surgery'
        ),
        'prob_exploratory_laparotomy': Parameter(
            Types.REAL,
            'probability that someone with an internal organ injury will require a exploratory_laparotomy'
        ),
        'prob_death_fractures_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a fracture injury will die without treatment'
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
        'rt_injury_1': Property(Types.CATEGORICAL, 'Codes for injury 1 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_2': Property(Types.CATEGORICAL, 'Codes for injury 2 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_3': Property(Types.CATEGORICAL, 'Codes for injury 3 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_4': Property(Types.CATEGORICAL, 'Codes for injury 4 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_5': Property(Types.CATEGORICAL, 'Codes for injury 5 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_6': Property(Types.CATEGORICAL, 'Codes for injury 6 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_7': Property(Types.CATEGORICAL, 'Codes for injury 7 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
        'rt_injury_8': Property(Types.CATEGORICAL, 'Codes for injury 8 from RTI',
                                categories=['none', '112', '113', '133', '134', '135', '211', '212', '2101', '291',
                                            '342', '343', '361', '363', '322', '323', '412', '414', '461', '463',
                                            '453', '441', '442', '443', '552', '553', '554', '612', '673', '674',
                                            '675', '676', '712', '722', '782', '783', '7101', '811', '812', '813',
                                            '822', '882', '883', '884', '8101']),
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

        df.loc[df.is_alive, 'rt_road_traffic_inc'] = False
        df.loc[df.is_alive, 'rt_injseverity'] = "none"  # default: no one has been injured in a RTI
        df.loc[df.is_alive, 'rt_injury_1'] = "none"
        df.loc[df.is_alive, 'rt_injury_2'] = "none"
        df.loc[df.is_alive, 'rt_injury_3'] = "none"
        df.loc[df.is_alive, 'rt_injury_4'] = "none"
        df.loc[df.is_alive, 'rt_injury_5'] = "none"
        df.loc[df.is_alive, 'rt_injury_6'] = "none"
        df.loc[df.is_alive, 'rt_injury_7'] = "none"
        df.loc[df.is_alive, 'rt_injury_8'] = "none"
        df.loc[df.is_alive, 'rt_polytrauma'] = False
        df.loc[df.is_alive, 'rt_perm_disability'] = False
        df.loc[df.is_alive, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.loc[df.is_alive, 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.loc[df.is_alive, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.loc[df.is_alive, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.loc[df.is_alive, 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_date_inj'] = pd.NaT

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
        df.at[child_id, 'rt_road_traffic_inc'] = False
        df.at[child_id, 'rt_injseverity'] = "none"  # default: no one has been injured in a RTI
        df.at[child_id, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.at[child_id, 'rt_injury_1'] = "none"
        df.at[child_id, 'rt_injury_2'] = "none"
        df.at[child_id, 'rt_injury_3'] = "none"
        df.at[child_id, 'rt_injury_4'] = "none"
        df.at[child_id, 'rt_injury_5'] = "none"
        df.at[child_id, 'rt_injury_6'] = "none"
        df.at[child_id, 'rt_injury_7'] = "none"
        df.at[child_id, 'rt_injury_8'] = "none"
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
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']
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
        now = self.sim.date

        # Reset injury properties after death
        immdeathidx = df.index[df.is_alive & df.rt_imm_death]
        deathwithmedidx = df.index[df.is_alive & df.rt_post_med_death]
        diedfromrtiidx = immdeathidx.union(deathwithmedidx)
        df.loc[diedfromrtiidx, "rt_imm_death"] = False
        df.loc[diedfromrtiidx, "rt_post_med_death"] = False
        df.loc[diedfromrtiidx, "rt_disability"] = 0
        df.loc[diedfromrtiidx, "rt_med_int"] = False
        df.loc[diedfromrtiidx, "rt_polytrauma"] = False
        df.loc[diedfromrtiidx, "rt_injseverity"] = "none"
        df.loc[diedfromrtiidx, "rt_perm_disability"] = False
        df.loc[diedfromrtiidx, "rt_injury_1"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_2"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_3"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_4"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_5"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_6"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_7"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_8"] = "none"
        df.loc[diedfromrtiidx, "rt_date_inj"] = pd.NaT
        # reset whether they have been selected for an injury this month
        df['rt_road_traffic_inc'] = False
        # reset whether they have sought care this month
        df['rt_med_int'] = False
        df.loc[df.is_alive, 'rt_post_med_death'] = False

        # --------------------------------- UPDATING OF RTI OVER TIME -------------------------------------------------
        rt_current_non_ind = df.index[df.is_alive & ~df.rt_road_traffic_inc & ~df.rt_imm_death]

        # ========= Update for people currently not involved in a RTI, make some involved in a RTI event ==============
        eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                         self.base_1m_prob_rti,
                         Predictor('sex').when('M', self.rr_injrti_male),
                         Predictor('age_years').when('.between(18,29)', self.rr_injrti_age1829),
                         Predictor('age_years').when('.between(30,39)', self.rr_injrti_age3039),
                         Predictor('age_years').when('.between(40,49)', self.rr_injrti_age4049),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(df.iloc[rt_current_non_ind])
        random_draw_in_rti = self.module.rng.random_sample(size=len(rt_current_non_ind))
        selected_for_rti = rt_current_non_ind[pred > random_draw_in_rti]

        # ========================= Take those involved in a RTI and assign some to death ==============================
        # Update to say they have been involved in a rti

        df.loc[selected_for_rti, 'rt_road_traffic_inc'] = True
        df.loc[selected_for_rti, 'rt_date_inj'] = now
        idx = df.index[df.is_alive & df.rt_road_traffic_inc]
        selected_to_die = idx[self.imm_death_proportion_rti > self.module.rng.random_sample(size=len(idx))]
        df.loc[selected_to_die, 'rt_imm_death'] = True

        for individual_id in selected_to_die:
            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, individual_id, "RTI_imm_death"),
                self.sim.date
            )

        # ============= Take those remaining people involved in a RTI and assign injuries to them ==================

        selected_for_rti_inj = df.loc[df.is_alive].copy()
        selected_for_rti_inj = selected_for_rti_inj.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death]

        mortality, description = injrandomizer(len(selected_for_rti_inj))
        description = description.replace('nan', 'none')
        description = description.set_index(selected_for_rti_inj.index)

        selected_for_rti_inj = selected_for_rti_inj.join(mortality.set_index(selected_for_rti_inj.index))

        selected_for_rti_inj = selected_for_rti_inj.join(description.set_index(selected_for_rti_inj.index))

        for ninjuries in range(0, len(description.columns)):
            for person_id in selected_for_rti_inj.index:
                if ninjuries == 0:
                    df.loc[person_id, 'rt_injury_1'] = description.loc[person_id, 'Injury 1']
                if ninjuries == 1:
                    df.loc[person_id, 'rt_injury_2'] = description.loc[person_id, 'Injury 2']
                if ninjuries == 2:
                    df.loc[person_id, 'rt_injury_3'] = description.loc[person_id, 'Injury 3']
                if ninjuries == 3:
                    df.loc[person_id, 'rt_injury_4'] = description.loc[person_id, 'Injury 4']
                if ninjuries == 4:
                    df.loc[person_id, 'rt_injury_5'] = description.loc[person_id, 'Injury 5']
                if ninjuries == 5:
                    df.loc[person_id, 'rt_injury_6'] = description.loc[person_id, 'Injury 6']
                if ninjuries == 6:
                    df.loc[person_id, 'rt_injury_7'] = description.loc[person_id, 'Injury 7']
                if ninjuries == 7:
                    df.loc[person_id, 'rt_injury_8'] = description.loc[person_id, 'Injury 8']
        # ============================ Injury severity classification============================================

        # ============================== Non specific injury updates ===============================================
        # Find those with mild injuries and update the rt_roadtrafficinj property so they have a mild injury
        mild_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj.is_alive & selected_for_rti_inj['ISS'] < 15]
        df.loc[mild_rti_idx, 'rt_injseverity'] = 'mild'
        # Find those with severe injuries and update the rt_roadtrafficinj property so they have a severe injury
        severe_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj['ISS'] >= 15]
        df.loc[severe_rti_idx, 'rt_injseverity'] = 'severe'
        # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
        polytrauma_idx = selected_for_rti_inj.index[selected_for_rti_inj['Polytrauma'] is True]
        df.loc[polytrauma_idx, 'rt_polytrauma'] = True

        # =+=+=+=+=+=+=+=+=+=+=+=+=+=+ Injury specific updates =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        # =+=+=+=+=+=+=+=+=+=+=+=+=+=+ Assign the DALY weights =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        # =============================== AIS region 1: head ==========================================================
        # ------ Find those with skull fractures and update rt_fracture to match and call for treatment ---------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('112').any(), axis=1)
        if len(inj1) > 0:
            idx1 = inj1.index[inj1]
            for injuredperson in idx1:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_unspecified_skull_fracture
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('113').any(), axis=1)
        if len(inj2) > 0:
            idx2 = inj2.index[inj2]
            for injuredperson in idx2:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_basilar_skull_fracture
        # ------ Find those with traumatic brain injury and update rt_tbi to match and call the TBI treatment ---------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('133').any(), axis=1)
        dalyweightsfor133 = [self.daly_wt_subarachnoid_hematoma, self.daly_wt_brain_contusion,
                             self.daly_wt_intraventricular_haemorrhage, self.daly_wt_subgaleal_hematoma]
        probabilities = [0.2, 0.66, 0.03, 0.11]
        if len(inj1) > 0:
            idx1 = inj1.index[inj1]
            for injuredperson in idx1:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor133,
                                                                                 p=probabilities)
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('134').any(), axis=1)
        dalyweightsfor134 = [self.daly_wt_epidural_hematoma, self.daly_wt_subdural_hematoma]
        probabilities = [0.52, 0.48]
        if len(inj2) > 0:
            idx2 = inj2.index[inj2]
            for injuredperson in idx2:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor134, p=probabilities)

        inj3 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('135').any(), axis=1)
        if len(inj3) > 0:
            idx3 = inj3.index[inj3]
            df.loc[idx3, 'rt_disability'] += self.daly_wt_diffuse_axonal_injury

        # =============================== AIS region 2: face ==========================================================
        # ----------------------- Find those with facial fractures and assign DALY weight -----------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('211').any(), axis=1)
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('212').any(), axis=1)
        if len(inj1) + len(inj2) > 0:
            idx1 = inj1.index[inj1]
            idx2 = inj2.index[inj2]
            idx = idx1.union(idx2)
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_facial_fracture

        # ----------------- Find those with lacerations/soft tissue injuries and assign DALY weight -------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('2101').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        # ----------------- Find those with eye injuries and assign DALY weight ---------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('291').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_eye_injury

        # =============================== AIS region 3: Neck ==========================================================
        # -------------------------- soft tissue injuries and internal bleeding----------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('342').any(), axis=1)
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('343').any(), axis=1)
        inj3 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('361').any(), axis=1)
        inj4 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('363').any(), axis=1)
        if len(inj1) + len(inj2) + len(inj3) + len(inj4) > 0:
            idx1 = inj1.index[inj1]
            idx2 = inj2.index[inj2]
            idx3 = inj3.index[inj3]
            idx4 = inj4.index[inj4]
            idx = idx1.union(idx2)
            idx = idx.union(idx3)
            idx = idx.union(idx4)
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_internal_bleeding

        # -------------------------------- neck vertebrae dislocation ------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('322').any(), axis=1)
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('323').any(), axis=1)
        if len(inj1) + len(inj2) > 0:
            idx1 = inj1.index[inj1]
            idx2 = inj2.index[inj2]
            idx = idx1.union(idx2)
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_dislocation

        # ================================== AIS region 4: Thorax =====================================================
        # --------------------------------- fractures & flail chest ---------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('412').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_rib_fracture
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('414').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_flail_chest
        # ------------------------------------ Internal bleeding ------------------------------------------------------
        # chest wall bruises/hematoma
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('461').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_chest_wall_bruises_hematoma
        # hemothorax
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('463').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_hemothorax
        # -------------------------------- Internal organ injury ------------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('453').any(), axis=1)
        dalyweightsfor453 = [self.daly_wt_diaphragm_rupture, self.daly_wt_lung_contusion]
        probabilities = [0.77, 0.23]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor453, p=probabilities)
        # ----------------------------------- Soft tissue injury ------------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('441').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_chest_wall_laceration
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('442').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_surgical_emphysema
        # ---------------------------------- Pneumothoraxs ------------------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('441').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_closed_pneumothorax
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('443').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_open_pneumothorax

        # ================================== AIS region 5: Abdomen ====================================================
        # Intestine, stomache and colon
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('552').any(), axis=1)
        # Spleen, Urinary bladder, Liver, Urethra, Diaphragm
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('553').any(), axis=1)
        # Kidney
        inj3 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('554').any(), axis=1)

        if len(inj1) + len(inj2) + len(inj3) > 0:
            idx1 = inj1.index[inj1]
            idx2 = inj2.index[inj2]
            idx3 = inj2.index[inj3]
            idx = idx1.union(idx2)
            idx = idx.union(idx3)
            df.loc[idx, 'rt_disability'] += self.daly_wt_abd_internal_organ_injury

        # =================================== AIS region 6: spine =====================================================
        # ----------------------------------- vertebrae fracture ------------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('612').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_vertebrae_fracture
        # ---------------------------------- Spinal cord injuries -----------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('673').any(), axis=1)
        dalyweightsfor673 = [self.daly_wt_spinal_cord_lesion_neck_without_treatment,
                             self.daly_wt_spinal_cord_lesion_below_neck_without_treatment]
        probabilities = [0.28, 0.72]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor673,
                                                                                 p=probabilities)

        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('674').any(), axis=1)
        inj2 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('675').any(), axis=1)

        dalyweightsfor674675 = [self.daly_wt_spinal_cord_lesion_neck_without_treatment,
                                self.daly_wt_spinal_cord_lesion_below_neck_without_treatment]
        probabilities = [0.39, 0.61]
        if len(inj1) + len(inj2) > 0:
            idx1 = inj1.index[inj1]
            idx2 = inj2.index[inj2]
            idx = idx1.union(idx2)
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor674675,
                                                                                 p=probabilities)

        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('676').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_neck_without_treatment

        # ============================== AIS body region 7: upper extremities ======================================
        # ------------------------------------------ fractures ------------------------------------------------------
        # Fracture to Clavicle, scapula, humerus, Hand/wrist, Radius/ulna
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('712').any(), axis=1)
        dalyweightsfor712 = [self.daly_wt_clavicle_scapula_humerus_fracture,
                             self.daly_wt_hand_wrist_fracture_without_treatment,
                             self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment]
        probabilities = [0.22, 0.59, 0.19]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor712,
                                                                                 p=probabilities)
        # ------------------------------------ Dislocation of shoulder ---------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('722').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_dislocated_shoulder
        # ------------------------------------------ Amputations -----------------------------------------------------
        # Amputation of fingers, Unilateral upper limb amputation, Thumb amputation
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('782').any(), axis=1)
        dalyweightsfor782 = [self.daly_wt_amputated_finger,
                             self.daly_wt_unilateral_arm_amputation_without_treatment,
                             self.daly_wt_amputated_thumb]
        probabilities = [0.66, 0.09, 0.25]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor782,
                                                                                 p=probabilities)
        # Bilateral upper limb amputation
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('783').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_arm_amputation_without_treatment
        # ----------------------------------- cuts and bruises --------------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('7101').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        # ============================== AIS body region 8: Lower extremities ========================================
        # ------------------------------------------ Fractures -------------------------------------------------------
        # Broken foot
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('811').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_foot_fracture_short_term_with_without_treatment
        # Broken patella, tibia, fibula
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('812').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_patella_tibia_fibula_fracture_with_treatment
        # Broken Hip, Pelvis, Femur other than femoral neck
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('813').any(), axis=1)
        dalyweightsfor813 = [self.daly_wt_hip_fracture_short_term_with_without_treatment,
                             self.daly_wt_pelvis_fracture_short_term,
                             self.daly_wt_femur_fracture_short_term]
        probabilities = [0.2, 0.2, 0.6]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor813,
                                                                                 p=probabilities)
        # -------------------------------------- Dislocations -------------------------------------------------------
        # Dislocated hip, knee
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('822').any(), axis=1)
        dalyweightsfor822 = [self.daly_wt_dislocated_hip,
                             self.daly_wt_dislocated_knee]
        probabilities = [0.94, 0.06]
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.module.rng.choice(dalyweightsfor822,
                                                                                 p=probabilities)
        # --------------------------------------- Amputations ------------------------------------------------------
        # toes
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('882').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_amputated_toes
        # Unilateral lower limb amputation
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('883').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_unilateral_lower_limb_amputation_without_treatment
        # Bilateral lower limb amputation
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('884').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_lower_limb_amputation_without_treatment
        # ------------------------------------ cuts and bruises -----------------------------------------------------
        inj1 = selected_for_rti_inj.apply(lambda row: row.astype(str).str.contains('8101').any(), axis=1)
        if len(inj1) > 0:
            idx = inj1.index[inj1]
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        DALYweightoverlimit = df.index[df['rt_disability'] > 1]
        df.loc[DALYweightoverlimit, 'rt_disability'] = 1

        idx = df.index[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death]
        for person_id_to_start_treatment in idx:
            event = HSI_RTI_MedicalIntervention(self.module, person_id=person_id_to_start_treatment)
            target_date = self.sim.date + DateOffset(days=int(0))
            self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0, topen=target_date,
                                                                tclose=None)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_RTI_MedicalIntervention(HSI_Event, IndividualScopeEventMixin):
    # todo: 1) Find way to schedule multiple diagnostic scans and treatments in this one HSI event.
    #  2) Find out the proportions of needing surgery for the various injuries
    #  3) Finish off the appointment requirements for amputations
    #  4) Include duration of stay, could be based on ISS score or at least influenced by it
    #  5) Include injury specific mortality for not having treatment
    #
    """This is a Health System Interaction Event.
    An appointment of a person who has experienced a road traffic injury, requiring the resources
    found at a level 1+ facility depending on the nature of the injury

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, RTI)
        df = self.sim.population.props
        p = module.parameters
        self.prob_TBI_require_craniotomy = p['prob_TBI_require_craniotomy']
        self.prob_exploratory_laparotomy = p['prob_exploratory_laparotomy']
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']
        self.prob_death_TBI_SCI_no_treatment = p['prob_death_TBI_SCI_no_treatment']
        self.prob_death_fractures_no_treatment = p['prob_death_fractures_no_treatment']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']
        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' foot
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        # ------------------------------------ Generic ----------------------------------------------------------------
        the_appt_footprint['AccidentsandEmerg'] = 1
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]
        surgery_counts = 0
        xray_counts = 0
        # print(persons_injuries)
        # --------------------- For fractures, sometimes requiring surgery ---------------------------------------------
        # ------------------------------- Skull fractures -------------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('112').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('113').any(), axis=1)
        idx2 = inj2.index[inj2]
        idx = idx1.union(idx2)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
            xray_counts += 1
        # -------------------------------- Facial fractures -----------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('211').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('212').any(), axis=1)
        idx2 = inj2.index[inj2]
        idx = idx1.union(idx2)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
            xray_counts += 1
        # --------------------------------- Thorax fractures ----------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('412').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('414').any(), axis=1)
        idx2 = inj2.index[inj2]
        idx = idx1.union(idx2)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
            xray_counts += 1
        # --------------------------------- Vertebrae fractures -------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('612').any(), axis=1)
        idx1 = inj1.index[inj1]
        if len(idx1) > 0:
            the_appt_footprint['DiagRadio'] = 1
            xray_counts += 1
        # --------------------------------- Upper extremity fractures --------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('712').any(), axis=1)
        idx1 = inj1.index[inj1]
        if len(idx1) > 0:
            the_appt_footprint['DiagRadio'] = 1
            xray_counts += 1
        # --------------------------------- Lower extremity fractures --------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('811').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('812').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('813').any(), axis=1)
        idx3 = inj3.index[inj3]
        idx = idx1.union(idx2).union(idx3)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray
            xray_counts += 1
        if len(idx3) > 0:
            the_appt_footprint['MajorSurg'] = 1
            surgery_counts += 1

        # ------------------------------ Traumatic brain injury requirements ------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('133').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('134').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('135').any(), axis=1)
        idx3 = inj3.index[inj3]
        idx = idx1.union(idx2).union(idx3)
        require_surgery = self.module.rng.random_sample(size=1)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['MRI'] = 1  # This appointment requires a MRI scan
            if require_surgery < self.prob_TBI_require_craniotomy:
                the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
                surgery_counts += 1
        # ------------------------------ Abdominal organ injury requirements ------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('552').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('553').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('554').any(), axis=1)
        idx3 = inj3.index[inj3]
        idx = idx1.union(idx2).union(idx3)
        require_surgery = self.module.rng.random_sample(size=1)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            if require_surgery < self.prob_exploratory_laparotomy:
                the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
                surgery_counts += 1
        # -------------------------------- Spinal cord injury requirements -------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('673').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('674').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('675').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('676').any(), axis=1)
        idx4 = inj4.index[inj4]
        idx = idx1.union(idx2).union(idx3).union(idx4)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray
            the_appt_footprint['MRI'] = 1  # This appointment requires an MRI
            the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
            xray_counts += 1
            surgery_counts += 1

        # --------------------------------- Dislocations ------------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('322').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('323').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('722').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('822').any(), axis=1)
        idx4 = inj4.index[inj4]
        idx = idx1.union(idx2).union(idx3).union(idx4)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray
            the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
            xray_counts += 1
            surgery_counts += 1

        # --------------------------------- Soft tissue injury in neck -------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('342').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('343').any(), axis=1)
        idx2 = inj2.index[inj2]
        idx = idx1.union(idx2)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
            surgery_counts += 1

        # --------------------------------- Soft tissue injury in thorax/ lung injury ----------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('441').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('443').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('453').any(), axis=1)
        idx3 = inj3.index[inj3]
        idx = idx1.union(idx2).union(idx3)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray
            the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
            xray_counts += 1
            surgery_counts += 1

        # -------------------------------- Internal bleeding -----------------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('361').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('363').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('461').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('463').any(), axis=1)
        idx4 = inj4.index[inj4]
        idx = idx1.union(idx2).union(idx3).union(idx4)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['MajorSurg'] = 1  # This appointment requires Major surgery
            surgery_counts += 1

        # ------------------------------------- Amputations ------------------------------------------------------------
        # Define the facilities at which this event can occur (only one is allowed)

        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_MedicalIntervention'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []
        # todo: 1) Find out whether it is general practice to perform all the needed surgeries all at once or perform
        #  them seperately.
        #  2) In regards to inpatient days, can I specify how long the injured person should stay in the health system
        #  for?
        #  3) Multiple injuries require multiple diagnostic tests, some can easily be done in the same resource i.e. ct
        #  scans, but if someone had multiple fractures and therefore needed multiple x rays, increasing the time
        #  consumed of the radiologist would this need to be reflected by scheduling multiple x ray sessions?

        # print('surgeries needed')
        # print(surgery_counts)
        # print('x rays needed')
        # print(xray_counts)

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props
        df.at[person_id, 'rt_med_int'] = True
        logger.debug('@@@@@@@@@@ Medical intervention started !!!!!!')
        # prob_dis = np.random.random()

        self.sim.schedule_event(RTIMedicalInterventionDeathEvent(self.module, person_id), self.sim.date +
                                DateOffset(days=0))
        logger.debug('This is MedicalInterventionEvent scheduling a potential death on date %s for person %d',
                     self.sim.date, person_id)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('HSI_RTI: did not run')
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('133').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('134').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('135').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('673').any(), axis=1)
        idx4 = inj4.index[inj4]
        inj5 = persons_injuries.apply(lambda row: row.astype(str).str.contains('674').any(), axis=1)
        idx5 = inj5.index[inj5]
        inj6 = persons_injuries.apply(lambda row: row.astype(str).str.contains('675').any(), axis=1)
        idx6 = inj6.index[inj6]
        inj7 = persons_injuries.apply(lambda row: row.astype(str).str.contains('676').any(), axis=1)
        idx7 = inj7.index[inj7]
        idx = idx1.union(idx2).union(idx3).union(idx4).union(idx5).union(idx6).union(idx7)
        if len(idx) > 0:
            prob_death_no_med = self.module.rng.random_sample(size=1)
            if prob_death_no_med < self.prob_death_TBI_SCI_no_treatment:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='death_without_med'), self.sim.date)
        #
        #
        #

        pass


class RTIMedicalInterventionDeathEvent(Event, IndividualScopeEventMixin):
    """This is the MedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent which determines the
    resources required to treat that person and
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']

    def apply(self, person_id):
        df = self.sim.population.props
        randfordeath = self.module.rng.random_sample(size=1)
        # Schedule death for those who died from their injuries despite medical intervention
        if df.loc[person_id, 'rt_injseverity'] == 'mild':
            if randfordeath < self.prob_death_with_med_mild:
                df.loc[person_id, 'rt_post_med_death'] = True
                print('death from mild')
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='death_with_med'), self.sim.date)
                # Log the death
                logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d on date %s',
                             person_id, self.sim.date)
            else:
                self.sim.schedule_event(RTIMedicalInterventionPermDisabilityEvent(self.module, person_id), self.sim.date
                                        + DateOffset(days=0))
        if df.loc[person_id, 'rt_injseverity'] == 'severe':
            if randfordeath < self.prob_death_with_med_severe:
                df.loc[person_id, 'rt_post_med_death'] = True
                print('death from severe')
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='death_with_med'), self.sim.date)
                # Log the death
                logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d on date %s',
                             person_id, self.sim.date)
            else:
                self.sim.schedule_event(RTIMedicalInterventionPermDisabilityEvent(self.module, person_id), self.sim.date
                                        + DateOffset(days=0))


class RTIMedicalInterventionPermDisabilityEvent(Event, IndividualScopeEventMixin):
    """This is the MedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent which determines the
    resources required to treat that person and
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)
        p = self.module.parameters
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']

    def apply(self, person_id):
        df = self.sim.population.props
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]
        # ------------------------ Track permanent disabilities with treatment ----------------------------------------
        # --------------------------------- Perm disability from TBI --------------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('133').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('134').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('135').any(), axis=1)
        idx3 = inj3.index[inj3]
        idx = idx1.union(idx2).union(idx3)
        if len(idx) > 0:
            prob_perm_disability = self.module.rng.random_sample(size=1)
            if prob_perm_disability < self.prob_perm_disability_with_treatment_severe_TBI:
                # print('person perm disabled TBI')
                # print(person_id)
                df.at[person_id, 'rt_perm_disability'] = True
                logger.debug('@@@@@@@@@@ Medical intervention for TBI started but still disabled!!!!!!')

        # ------------------------------------- Perm disability from SCI ----------------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('673').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('674').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('675').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('676').any(), axis=1)
        idx4 = inj3.index[inj4]
        idx = idx1.union(idx2).union(idx3).union(idx4)
        if len(idx) > 0:
            prob_perm_disability = self.module.rng.random_sample(size=1)
            if prob_perm_disability < self.prob_perm_disability_with_treatment_sci:
                # print('person perm disabled SCI')
                df.at[person_id, 'rt_perm_disability'] = True
                logger.debug('@@@@@@@@@@ Medical intervention for SCI started but still disabled!!!!!!')
        # ------------------------------------- Perm disability from amputation ----------------------------------------
        inj1 = persons_injuries.apply(lambda row: row.astype(str).str.contains('782').any(), axis=1)
        idx1 = inj1.index[inj1]
        inj2 = persons_injuries.apply(lambda row: row.astype(str).str.contains('783').any(), axis=1)
        idx2 = inj2.index[inj2]
        inj3 = persons_injuries.apply(lambda row: row.astype(str).str.contains('882').any(), axis=1)
        idx3 = inj3.index[inj3]
        inj4 = persons_injuries.apply(lambda row: row.astype(str).str.contains('883').any(), axis=1)
        idx4 = inj4.index[inj4]
        inj5 = persons_injuries.apply(lambda row: row.astype(str).str.contains('884').any(), axis=1)
        idx5 = inj5.index[inj5]
        idx = idx1.union(idx2).union(idx3).union(idx4).union(idx5)
        if len(idx) > 0:
            df.at[person_id, 'rt_perm_disability'] = True
            logger.debug('@@@@@@@@@@ Medical intervention for amputation still disabled!!!!!!')


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

        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, RTI)
        self.tot1inj = 0
        self.tot2inj = 0
        self.tot3inj = 0
        self.tot4inj = 0
        self.tot5inj = 0
        self.tot6inj = 0
        self.tot7inj = 0
        self.tot8inj = 0
        self.totfracnumber = 0
        self.totdisnumber = 0
        self.tottbi = 0
        self.totsoft = 0
        self.totintorg = 0
        self.totintbled = 0
        self.totsci = 0
        self.totamp = 0
        self.toteye = 0
        self.totextlac = 0
        self.totAIS1 = 0
        self.totAIS2 = 0
        self.totAIS3 = 0
        self.totAIS4 = 0
        self.totAIS5 = 0
        self.totAIS6 = 0
        self.totAIS7 = 0
        self.totAIS8 = 0
        self.totmild = 0
        self.totsevere = 0
        self.totinjured = 0
        self.deathonscene = 0
        self.soughtmedcare = 0
        self.deathaftermed = 0
        self.permdis = 0

    def apply(self, population):
        # Make some summary statitics
        df = population.props
        n_in_RTI = (df.rt_road_traffic_inc).sum()
        self.totinjured += n_in_RTI
        n_perm_disabled = (df.is_alive & df.rt_perm_disability).sum()
        # self.permdis += n_perm_disabled
        n_alive = df.is_alive.sum()
        n_not_injured = (df.is_alive & ~df.rt_road_traffic_inc).sum()
        n_immediate_death = (df.rt_road_traffic_inc & df.rt_imm_death).sum()
        self.deathonscene += n_immediate_death
        n_sought_care = (df.rt_road_traffic_inc & df.rt_med_int).sum()
        self.soughtmedcare += n_sought_care
        n_death_post_med = (df.is_alive & df.rt_post_med_death).sum()
        self.deathaftermed += n_death_post_med

        # n_head_injuries = (df.is_alive & df.rt_tbi).sum()

        def find_and_count_injuries(dataframe, tloinjcodes):
            index = pd.Index([])
            counts = 0
            for code in tloinjcodes:
                inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(), axis=1)
                if len(inj) > 0:
                    injidx = inj.index[inj]
                    counts += len(injidx)
                    index = index.union(injidx)
            return index, counts

        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        thoseininjuries = df.loc[df.rt_road_traffic_inc]
        df_injuries = thoseininjuries.loc[:, columns]
        # ==================================== Number of injuries =====================================================
        oneinjury = len(df_injuries.loc[df_injuries['rt_injury_2'] == 'none'])
        self.tot1inj += oneinjury
        twoinjury = len(df_injuries.loc[(df_injuries['rt_injury_2'] != 'none') &
                                        (df_injuries['rt_injury_3'] == 'none')])
        self.tot2inj += twoinjury
        threeinjury = len(df_injuries.loc[(df_injuries['rt_injury_3'] != 'none') &
                                          (df_injuries['rt_injury_4'] == 'none')])
        self.tot3inj += threeinjury
        fourinjury = len(df_injuries.loc[(df_injuries['rt_injury_4'] != 'none') &
                                         (df_injuries['rt_injury_5'] == 'none')])
        self.tot4inj += fourinjury
        fiveinjury = len(df_injuries.loc[(df_injuries['rt_injury_5'] != 'none') &
                                         (df_injuries['rt_injury_6'] == 'none')])
        self.tot5inj += fiveinjury
        sixinjury = len(df_injuries.loc[(df_injuries['rt_injury_6'] != 'none') &
                                        (df_injuries['rt_injury_7'] == 'none')])
        self.tot6inj += sixinjury
        seveninjury = len(df_injuries.loc[(df_injuries['rt_injury_7'] != 'none') &
                                          (df_injuries['rt_injury_8'] == 'none')])
        self.tot7inj += seveninjury
        eightinjury = len(df_injuries.loc[df_injuries['rt_injury_8'] != 'none'])
        self.tot8inj += eightinjury
        # ====================================== AIS body regions =====================================================
        AIS1codes = ['112', '113', '133', '134', '135']
        AIS2codes = ['211', '212', '2101', '291']
        AIS3codes = ['342', '343', '361', '363', '322', '323']
        AIS4codes = ['412', '414', '461', '463', '453', '441', '442', '443']
        AIS5codes = ['552', '553', '554']
        AIS6codes = ['612', '673', '674', '675', '676']
        AIS7codes = ['712', '722', '782', '783', '7101']
        AIS8codes = ['811', '812', '813', '822', '882', '883', '884', '8101']
        idx, AIS1counts = find_and_count_injuries(df_injuries, AIS1codes)
        self.totAIS1 += AIS1counts
        idx, AIS2counts = find_and_count_injuries(df_injuries, AIS2codes)
        self.totAIS2 += AIS2counts
        idx, AIS3counts = find_and_count_injuries(df_injuries, AIS3codes)
        self.totAIS3 += AIS3counts
        idx, AIS4counts = find_and_count_injuries(df_injuries, AIS4codes)
        self.totAIS4 += AIS4counts
        idx, AIS5counts = find_and_count_injuries(df_injuries, AIS5codes)
        self.totAIS5 += AIS5counts
        idx, AIS6counts = find_and_count_injuries(df_injuries, AIS6codes)
        self.totAIS6 += AIS6counts
        idx, AIS7counts = find_and_count_injuries(df_injuries, AIS7codes)
        self.totAIS7 += AIS7counts
        idx, AIS8counts = find_and_count_injuries(df_injuries, AIS8codes)
        self.totAIS8 += AIS8counts
        # ================================== Injury characteristics ===================================================

        allfraccodes = ['112', '113', '211', '212', '412', '414', '612', '712', '811', '812', '813']
        idx, fraccounts = find_and_count_injuries(df_injuries, allfraccodes)
        self.totfracnumber += fraccounts
        dislocationcodes = ['322', '323', '722', '822']
        idx, dislocationcounts = find_and_count_injuries(df_injuries, dislocationcodes)
        self.totdisnumber += dislocationcounts
        allheadinjcodes = ['133', '134', '135']
        idx, tbicounts = find_and_count_injuries(df_injuries, allheadinjcodes)
        self.tottbi += tbicounts
        softtissueinjcodes = ['342', '343', '441', '442', '443']
        idx, softtissueinjcounts = find_and_count_injuries(df_injuries, softtissueinjcodes)
        self.totsoft += softtissueinjcounts
        organinjurycodes = ['453', '552', '553', '554']
        idx, organinjurycounts = find_and_count_injuries(df_injuries, organinjurycodes)
        self.totintorg += organinjurycounts
        internalbleedingcodes = ['361', '363', '461', '463']
        idx, internalbleedingcounts = find_and_count_injuries(df_injuries, internalbleedingcodes)
        self.totintbled += internalbleedingcounts
        spinalcordinjurycodes = ['673', '674', '675', '676']
        idx, spinalcordinjurycounts = find_and_count_injuries(df_injuries, spinalcordinjurycodes)
        self.totsci += spinalcordinjurycounts
        amputationcodes = ['782', '783', '882', '883', '884']
        idx, amputationcounts = find_and_count_injuries(df_injuries, amputationcodes)
        self.totamp += amputationcounts
        eyecodes = ['291']
        idx, eyecounts = find_and_count_injuries(df_injuries, eyecodes)
        self.toteye += eyecounts
        externallacerationcodes = ['2101', '7101', '8101']
        idx, externallacerationcounts = find_and_count_injuries(df_injuries, externallacerationcodes)
        self.totextlac += externallacerationcounts
        totalinj = fraccounts + dislocationcounts + tbicounts + softtissueinjcounts + organinjurycounts + \
                   internalbleedingcounts + spinalcordinjurycounts + amputationcounts + externallacerationcounts

        # ================================= Injury severity ===========================================================
        sev = df.loc[df.rt_road_traffic_inc]
        sev = sev['rt_injseverity']
        severity, severitycount = np.unique(sev, return_counts=True)
        if 'mild' in severity:
            idx = np.where(severity == 'mild')
            self.totmild += severitycount[idx]
        if 'severe' in severity:
            idx = np.where(severity == 'severe')
            self.totsevere += severitycount[idx]

        dict_to_output = {
            'number involved in a rti': n_in_RTI,
            'number not injured': n_not_injured,
            'number alive': n_alive,
            'number immediate deaths': n_immediate_death,
            'number deaths post med': n_death_post_med,
            # 'number head injuries': n_head_injuries,
            'number permanently disabled': n_perm_disabled,
            'total injuries': totalinj,
            # 'proportion fractures': fraccounts / totalinj,
            # 'proportion dislocations': dislocationcounts / totalinj,
            # 'proportion tbi': tbicounts / totalinj,
            # 'proportion soft tissue injuries': softtissueinjcounts / totalinj,
            # 'proportion organ injuries': organinjurycounts / totalinj,
            # 'proportion internal bleeding': internalbleedingcounts / totalinj,
            # 'proportion spinal cord injury': spinalcordinjurycounts / totalinj,
            # 'proportion amputations': amputationcounts / totalinj,
            # 'proportion external lacerations': externallacerationcounts / totalinj
            'number of fractures': fraccounts,
            'number of dislocations': dislocationcounts,
            'number of tbi': tbicounts,
            'number of soft tissue injuries': softtissueinjcounts,
            'number of organ injuries': organinjurycounts,
            'number of internal bleeding': internalbleedingcounts,
            'number of spinal cord injury': spinalcordinjurycounts,
            'number of amputations': amputationcounts,
            'number of eye injuries': eyecounts,
            'number of external lacerations': externallacerationcounts

        }
        # -------------------------------------- Stored outputs -------------------------------------------------------
        injcategories = [self.totfracnumber, self.totdisnumber, self.tottbi, self.totsoft, self.totintorg,
                         self.totintbled, self.totsci, self.totamp, self.toteye, self.totextlac]
        np.savetxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/Injcategories.txt', injcategories,
                   delimiter=',')
        injlocs = [self.totAIS1, self.totAIS2, self.totAIS3, self.totAIS4, self.totAIS5, self.totAIS6, self.totAIS7,
                   self.totAIS8]
        np.savetxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/Injlocs.txt', injlocs)
        injseverity = [self.totmild, self.totsevere]
        np.savetxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/Injsev.txt', injseverity)
        numberinjdist = [self.tot1inj, self.tot2inj, self.tot3inj, self.tot4inj, self.tot5inj, self.tot6inj,
                         self.tot7inj, self.tot8inj]
        np.savetxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/Injnumber.txt', numberinjdist)
        rtiflow = [self.totinjured, self.deathonscene, self.soughtmedcare, self.deathaftermed, n_perm_disabled]
        np.savetxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIflow.txt', rtiflow)

        logger.info('%s|summary_1m|%s', self.sim.date, dict_to_output)
