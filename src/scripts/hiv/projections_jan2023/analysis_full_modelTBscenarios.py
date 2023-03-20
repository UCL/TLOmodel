
def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=100))

    def apply(self, population):

        p = self.module.parameters
        scenario = p["scenario"]

        logger.debug(
            key="message", data=f"ScenarioSetupEvent: scenario {scenario}"
        )

 if scenario == 0:
            return

        # all scenarios 1-4 have scale-up of testing/treatment
        if scenario > 0:

        if scenario=1
            scenario=xpert
            self.sim.modules["HealthSystem"].parameters["cons_availability"]["xpert"]
             self.sim.modules["Tb"].parameters["sens_xpert_smear_negative"] =0.68
            self.sim.modules["Tb"].parameters["sens_xpert_smear_positive"] =0.98
            self.sim.modules["Tb"].parameters["spec_xpert_smear_positive"] =0.99
            self.sim.modules["Tb"].parameters["spec_xpert_smear_positive"] =0.99

            if scenario=2
            scenario = xray
            self.sim.modules["HealthSystem"].parameters["cons_availability"]["xray"]
            self.sim.modules["Tb"].parameters["sens__xray_smear_negative"] = 0.8
            self.sim.modules["Tb"].parameters["sens__xray_smear_positive"] = 0.91
            self.sim.modules["Tb"].parameters["spec__xray_smear_positive"] = 0.67
            self.sim.modules["Tb"].parameters["spec__xray_smear_positive"] = 0.67
            self.sim.modules["Tb"].parameters["probability_access_to_xray"] = 0.1

            if scenario=3
            scenario = sputum
            self.sim.modules["HealthSystem"].parameters["cons_availability"]["sputum"]
            self.sim.modules["Tb"].parameters["sens__sputum_smear_negative"] = 1.0
            self.sim.modules["Tb"].parameters["sens__sputum_smear_positive"] = 1.0

            if scenario=4
            scenario = Tboutreach
            self.sim.modules["HealthSystem"].parameters["cons_availability"]["xray"]
            # need to set facility level as outreach services use facilities from district
            self.sim.modules["Tb"].parameters["sens__xray_smear_negative"] = 0.8
            self.sim.modules["Tb"].parameters["sens__xray_smear_positive"] = 0.91
            self.sim.modules["Tb"].parameters["spec__xray_smear_positive"] = 0.67
            self.sim.modules["Tb"].parameters["spec__xray_smear_positive"] = 0.67
            self.sim.modules["Tb"].parameters["probability_community_chest_xray"] = 0.1
            self.sim.modules["Tb"].parameters["probability_access_to_xray"] = 0.1
