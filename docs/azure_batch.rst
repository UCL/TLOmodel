=================
Using Azure Batch
=================

Scenarios configure TLOmodel simulations. Simulations are configured as subclasses of
``tlo.scenario.BaseScenario``. The ``tlo`` command-line tool is used to submit simulation scenarios to run on
Azure Batch.

Setup
=====

1. `Install the Azure command line interface <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>`_
2. If you have not already done so ensure you have followed the instructions to set up a ``tlo`` `Conda <https://docs.conda.io/en/latest/>`_ environment in :doc:`readme`.
3. Open Terminal (MacOS) or Anaconda Prompt (Windows)
4. Activate the ``tlo`` environment by running ``conda activate tlo``
5. *Extra step for Windows only*
    * Run the following command::

        python %CONDA_PREFIX%\Scripts\pywin32_postinstall.py -install

    * If successful, there will be some output, ending with *"The pywin32 extensions were successfully installed."*
6. Change the working directory to the root directory of the ``TLOmodel`` repository
7. Copy the file ``tlo.example.conf`` to ``tlo.conf``, and fill in the fields as required (you will need to ask the team for the details here).
8. In the terminal check the following command runs: ``tlo``
9. Login to Azure with ``az login --tenant 1faf88fe-a998-4c5b-93c9-210a11d9a5c2``
10. In the terminal run: ``tlo batch-list``



Creating a Scenario
===================

1. Checkout a new branch for your scenario
2. Create a file ``src/scripts/<your directory>/<a unique filename>.py``. The filename will serve as the job identifier.
3. In the file, create a subclass of BaseScenario. e.g.::

    from tlo import Date
    from tlo import logging
    from tlo.scenario import BaseScenario
    from tlo.methods import demography, enhanced_lifestyle

    class MyTestScenario(BaseScenario):
        def __init__(self):
            super().__init__()
            self.seed = 12
            self.start_date = Date(2010, 1, 1)
            self.end_date = Date(2020, 1, 1)
            self.pop_size = 1000
            self.number_of_draws = 2
            self.runs_per_draw = 2

        def log_configuration(self):
            return {
                'filename': 'my_test_scenario', 'directory': './outputs',
                'custom_levels': {'*': logging.INFO}
            }

        def modules(self):
            return [
                demography.Demography(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            ]

        def draw_parameters(self, draw_number, rng):
            return {
                'Lifestyle': {
                    'init_p_urban': rng.randint(10, 20) / 100.0,
                }
            }

4. Check the batch configuration gets generated without error::

    tlo scenario-run --draw-only src/scripts/dev/tlo_q1_demo.py

5. Test the scenario starts running without problems::

    tlo scenario-run src/scripts/dev/tlo_q1_demo.py

   or execute a single run::

        tlo scenario-run src/scripts/dev/tlo_q1_demo.py --draw 1 0

6. Commit the scenario file and push to Github

Interacting with Azure Batch
----------------------------

Each of the subcommands of ``tlo`` has a ``--help`` flag for further information e.g. ``tlo batch-submit --help``

*Submit scenario to Azure Batch*::

    tlo batch-submit src/scripts/dev/tlo_q1_demo.py

*List jobs currently on Azure Batch where id contains "tamuri"*::

    tlo batch-list -f tlo_q1

*List active jobs (default 5 jobs listed)*::

    tlo batch-list --active

*Display information about a job*::

    tlo batch-job tlo_q1_demo-123 --tasks

*Download result files for a completed job*::

    tlo batch-download tlo_q1_demo-123

