"""
A script called from tables.py, containing a function
* plot_costs() which plots
 - the total costs per periods from the table, and
 - the total costs for the entire time shown in the table (to be used in presentations).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_costs(in_id, in_suffix, in_x_labels, in_cons_costs_without, in_cons_costs_with,
               in_pop_interv_costs_with, in_ppfp_interv_costs_with, in_mwk_to_usd_exchange_rate,
               in_reduce_magnitude=1e3):
    def rgb_perc(nmb_r, nmb_g, nmb_b):
        return nmb_r / 255, nmb_g / 255, nmb_b / 255
        """
        Converts 0-255 RGB colors to 0-1 RGB.
        """

    # Where will output fig go - by default, wherever this script is run
    outputpath = Path("./outputs")  # folder for convenience of storing outputs
    # output file name
    output_filename = str('Consumables and Interventions Costs ' + in_id[0] + "_" + in_id[1] + in_suffix + '.png')

    # width of the bars
    width = 0.3

    # define colors
    colors = {
        'cons': rgb_perc(244, 165, 130),  # consumables
        'pop': rgb_perc(146, 197, 222),  # pop intervention
        'ppfp': rgb_perc(5, 113, 176)  # ppfp intervention
    }

    # define labels
    labels = {
        'cons': 'consumables Without/With interventions',  # consumables
        'pop': 'Pop intervention implementation',  # pop intervention
        'ppfp': 'PPFP intervention implementation',  # ppfp intervention
        'y_axis_mwk': 'billions (1e9) MWK',  # y-axis label for costs in MWK
        'y_axis_usd': 'millions (1e6) USD'  # y-axis label costs in USD
    }

    # define title
    fig_title = 'Consumables & Interventions Costs'

    # Prepare data for the plots
    def reduce_magnitude(in_list, in_in_reduce_magnitude):
        return [x / in_in_reduce_magnitude for x in in_list]

    cons_costs_without = reduce_magnitude(in_cons_costs_without, in_reduce_magnitude)
    #
    cons_costs_with = reduce_magnitude(in_cons_costs_with, in_reduce_magnitude)
    pop_interv_costs_with = reduce_magnitude(in_pop_interv_costs_with, in_reduce_magnitude)
    ppfp_interv_costs_with = reduce_magnitude(in_ppfp_interv_costs_with, in_reduce_magnitude)
    ppfp_bottom = [x + y for x, y in zip(cons_costs_with, pop_interv_costs_with)]

    # %%% Plot all time periods + total ................................................................................
    fig, ax = plt.subplots()
    # custom x-axis tick labels
    x_labels_tp = in_x_labels.copy()
    x_labels_tp[-1] = "TOTAL (" + x_labels_tp[-1] + ")"
    # x labels position: i = 1st bar, i+w/2 = year, i+w = 2nd bar
    x_i = np.arange(len(x_labels_tp))  # the x_label locations
    x = list()
    for i in x_i:
        x.extend([i - 0.7 * width, i, i + width * 0.65])
    plt.xticks(x)
    x_labels_all = list()
    for x_label_y in x_labels_tp:
        x_labels_all.extend(["Without"])
        x_labels_all.extend([str("\n" + str(x_label_y))])
        x_labels_all.extend(["With"])
    ax.set_xticklabels(x_labels_all, ha='center')
    # hide tick lines for x-axis
    ax.tick_params(axis='x', which='both', length=0)
    # title
    ax.set_title(fig_title, fontweight="bold")
    # set y-axis label
    ax.set_ylabel(labels['y_axis_mwk'], fontweight="bold")
    # TODO: more values on y-axis

    # bar_without
    ax.bar(x_i - width * 0.6, cons_costs_without, width,
           label=labels['cons'], color=colors['cons'])
    # bar_with
    ax.bar(x_i + width * 0.6, cons_costs_with, width, color=colors['cons'])
    # bar_with_pop_interv
    ax.bar(x_i + width * 0.6, pop_interv_costs_with, width, bottom=cons_costs_with,
           label=labels['pop'], color=colors['pop'])
    # bar_with_ppfp_interv
    ax.bar(x_i + width * 0.6, ppfp_interv_costs_with, width, bottom=ppfp_bottom,
           label=labels['ppfp'], color=colors['ppfp'])
    # add legend
    ax.legend()
    plt.grid(axis='y')

    # the below needs at least 3.4 version of matplotlib package (we have 3.3.4)
    # ax.bar_label(bar_without, padding=3)
    # ax.bar_label(bar_with, padding=3)

    # add secondary y-axis labels
    ax_us = ax.secondary_yaxis("right")
    ax_us.set_ylabel(labels['y_axis_usd'], fontweight="bold")
    y_labels_usd = list()
    for y_label_mwk in ax.get_yticks():
        y_labels_usd.extend([str(round(y_label_mwk * in_mwk_to_usd_exchange_rate * 1000))])
    ax_us.set_yticks(ax.get_yticks())
    ax_us.set_yticklabels(y_labels_usd)

    fig.tight_layout()

    plt.savefig(outputpath / output_filename, format='png')

    # %%% Plot total only ..............................................................................................
    fig2, ax2 = plt.subplots()
    # custom x-axis tick labels
    x2_i = np.array(0)  # the x_label locations
    x2 = [0 - width * 0.2, 0, 0 + width * 0.2]
    plt.xticks(x2)
    ax2.set_xticklabels(['Without', str('\n' + in_x_labels[-1]), 'With'], ha='center')
    # hide tick lines for x-axis
    ax2.tick_params(axis='x', which='both', length=0)
    # title
    ax2.set_title(str('TOTAL\n' + fig_title), fontweight="bold", x=0.45)
    # y-axis label
    ax2.set_ylabel(labels['y_axis_mwk'], fontweight="bold")
    # bar_without
    ax2.bar(x2_i - width * 0.2, cons_costs_without[-1], width/3,
            label=labels['cons'], color=colors['cons'])
    # bar_with
    ax2.bar(x2_i + width * 0.2, cons_costs_with[-1], width/3,
            color=colors['cons'])
    # bar_with_pop_interv
    ax2.bar(x2_i + width * 0.2, pop_interv_costs_with[-1], width/3, bottom=cons_costs_with[-1],
            label=labels['pop'], color=colors['pop'])
    # bar_with_ppfp_interv
    ax2.bar(x2_i + width * 0.2, ppfp_interv_costs_with[-1], width/3, bottom=ppfp_bottom[-1],
            label=labels['ppfp'], color=colors['ppfp'])
    # add legend
    ax2.legend(loc='upper left', bbox_to_anchor=(0, -.12))
    plt.grid(axis='y')

    # add secondary y-axis labels
    ax2_us = ax2.secondary_yaxis("right")
    ax2_us.set_ylabel(labels['y_axis_usd'], fontweight="bold")
    y2_labels_usd = list()
    for y2_label_mwk in ax2.get_yticks():
        y2_labels_usd.extend([str(round(y2_label_mwk * in_mwk_to_usd_exchange_rate * 1000))])
    ax2_us.set_yticks(ax2.get_yticks())
    ax2_us.set_yticklabels(y2_labels_usd)

    fig2.tight_layout()

    plt.savefig(outputpath / (str('Total ' + output_filename)), format='png')

    print("Fig: Consumables and Interventions Costs Over time saved.")
