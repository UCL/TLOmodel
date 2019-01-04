#
#
# from tlo import Simulation, Date
# from tlo.methods import demography, pregnancy_and_newborn_supervisor
#
# Edit this path so it points to your own copy of the Demography.xlsx file
# path = 'Demography_WorkingFile.xlsx'
# start_date = Date(2010, 1, 1)
# end_date = Date(2060, 1, 1)
# popsize = 10
#
# sim = Simulation(start_date=start_date)
#
# core_module = demography.Demography(workbook_path=path)
#
# pregnancy_and_newborn_supervisor=pregnancy_and_newborn_supervisor.Pregnancy_And_Newborn_Supervisor()
#
# sim.register(core_module, pregnancy_and_newborn_supervisor)
#
# sim.seed_rngs(0)
#
# sim.make_initial_population(n=popsize)
# sim.simulate(end_date=end_date)
#
#
#
#
