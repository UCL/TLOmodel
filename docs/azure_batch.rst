=================
Using Azure Batch
=================

Scenarios configure TLOmodel simulations. Simulations are configured as subclasses of
``tlo.scenario.BaseScenario``. The ``tlo`` command-line tool is used to submit simulation scenarios to run on
Azure Batch.

Setup
=====

1. Install Azure CLI `<https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>`_
2. Checkout & pull TLOmodel master branch
3. Open Terminal (MacOS) or Anaconda Prompt (Windows)
4. Change directory to the TLOmodel repository
5. Activate the Conda environment: ``conda activate tlo38``
6. ``pip install -r requirements\base.txt``
7. ``pip install -e .``
8. Extra steps on Windows
    * Run the following command, changing the path to where the Conda environment is installed::

        python C:\Users\Public\miniconda3\envs\tlo\Scripts\pywin32_postinstall.py -install

    * There will be some output, ending with *"The pywin32 extensions were successfully installed."*
9. Copy the file ``tlo.example.com`` to ``tlo.conf``, and fill in the fields as required (you will need to ask the team for the details here).
10. In Terminal/Anaconda Prompt check the following command runs: ``tlo``
11. Login to Azure with ``az login --tenant 1faf88fe-a998-4c5b-93c9-210a11d9a5c2``
12. In Terminal/Anaconda Prompt run: ``tlo batch-list``



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

