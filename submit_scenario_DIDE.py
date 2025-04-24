import csv
from pathlib import Path
from string import Template

cmd = Template("hipercow task create -- tlo scenario-run --draw ${i} ${j} --output-dir outputs/effect_of_capabilities_scaling/${I}/${j} src/scripts/healthsystem/impact_of_const_capabilities_expansion/scenario_impact_of_capabilities_expansion_scaling.py")
cmds = [cmd.substitute({"i": i, "j": j}) for i in range(9) for j in range(5)]

cmds_path = Path('.outputs/effect_of_capabilities_scaling/effect_of_capabilities_scaling.csv')
with open(cmds_path, "w", newline="") as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar=' ')
    for item in cmds:
        writer.writerow([item])
