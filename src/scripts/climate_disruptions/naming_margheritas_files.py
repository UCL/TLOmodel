import csv
from pathlib import Path
from string import Template

cmd = Template("hipercow task create -- tlo scenario-run --draw ${i} ${j} --output-dir outputs/margherita_scenarios_expansion_capabilities/${i}/${j} src/scripts/healthsystem/impact_of_const_capabilities_expansion/scenario_impact_of_capabilities_expansion_scaling.py")
cmds = [cmd.substitute({"i": i, "j": j}) for i in range(40) for j in range(10)]

cmds_path = Path('/Volumes/TLO/remw/TLOmodel/outputs/margherita_scenarios_expansion_capabilities.csv')
with open(cmds_path, "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=' ')
    for item in cmds:
        print(item)
        writer.writerow([item])
