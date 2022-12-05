from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    CAUSE_OF_DEATH_LABEL_TO_COLOR_MAP,
    get_coarse_appt_type,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    get_filtered_treatment_ids,
    order_of_coarse_appt,
    order_of_short_treatment_ids,
)

PREFIX_ON_FILENAME = '0'


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Plot the legend for (Short) TREATMENT_ID and (Coarse) APPT_TYPE, which are used in the standard plots."""

    def plot_legend(labels: Iterable, colors: Iterable, title: str = None) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
        for i, (_label, _color) in enumerate(zip(labels, colors)):
            ax.bar(i, np.nan, color=_color, label=_label)
        ax.legend(fontsize=14, ncol=2, loc='center')
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        fig.savefig(output_folder / f"{PREFIX_ON_FILENAME}_{title.replace(' ', '_')}.png")
        return fig, ax

    # %% Short TREATMENT_ID
    x = get_filtered_treatment_ids(depth=1)

    short_treatment_ids = sorted(x, key=order_of_short_treatment_ids)

    fig, ax = plot_legend(
        labels=[_label for _label in short_treatment_ids],
        colors=[get_color_short_treatment_id(x) for x in short_treatment_ids],
        title="TREATMENT_ID (Short)",
    )
    fig.tight_layout()
    fig.show()
    plt.close(fig)

    # %% Coarse Appt Type
    coarse_appt_types = sorted(
        pd.read_csv(
            resourcefilepath / 'healthsystem' / 'human_resources' / 'definitions' / 'ResourceFile_Appt_Types_Table.csv'
        )['Appt_Type_Code'].map(get_coarse_appt_type).drop_duplicates().values,
        key=order_of_coarse_appt
    )

    fig, ax = plot_legend(
        labels=coarse_appt_types,
        colors=[get_color_coarse_appt(x) for x in coarse_appt_types],
        title="Appointment Types (Coarse)",
    )
    fig.show()
    plt.close(fig)

    # %% Cause of Death Labels
    fig, ax = plot_legend(
        labels=list(CAUSE_OF_DEATH_LABEL_TO_COLOR_MAP.keys()),
        colors=list(CAUSE_OF_DEATH_LABEL_TO_COLOR_MAP.values()),
        title="Cause-of-Death Labels",
    )
    fig.show()
    plt.close(fig)


if __name__ == "__main__":
    apply(
        results_folder=None,
        output_folder=Path('./outputs'),
        resourcefilepath=Path('./resources')
    )
