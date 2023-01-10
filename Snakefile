
"""``snakemake`` file that runs entire analysis."""

# Imports ---------------------------------------------------------------------
import glob
import itertools
import os.path
import os
import textwrap
import urllib.request

import Bio.SeqIO

import pandas as pd

# Configuration  --------------------------------------------------------------

configfile: 'config.yaml'

# Functions -------------------------------------------------------------------

def nb_markdown(nb):
    """Return path to Markdown results of notebook `nb`."""
    return os.path.join(config['summary_dir'],
                        os.path.basename(os.path.splitext(nb)[0]) + '.md')

# Target rules ---------------------------------------------------------------

localrules: all

rule all:
    input:
        'results/summary/virus_titers_Delta_serum_validation_mutants.md',
        'results/summary/spike_neutralization-Delta_sera.md',
        'results/summary/REGN10933_yeast_lenti_dms_comparison.md'
        


# Rules ---------------------------------------------------------------------


rule get_virus_titers:
    """calculate virus titers for functional mutants"""
    input:
        config['virus_titers_Delta_serum']
    output:
        nb_markdown=nb_markdown('virus_titers_Delta_serum_validation_mutants.ipynb')
    params:
        nb='virus_titers_Delta_serum_validation_mutants.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

rule plot_neuts_spike:
    """plot neut curves for spike pseudotyped virus"""
    input:
        depletion_neuts=config['mAb_neuts_Delta']
    output:
        nb_markdown=nb_markdown('spike_neutralization-Delta_sera.ipynb')
    params:
        nb='spike_neutralization-Delta_sera.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

rule compare_yeastDMS_vs_lentiDMS:
    """compare yeast and lentivirus DMS for Ly-CoV1404"""
    input:
        yeast_DMS=config['yeast_dms_REGN10933_Star'],
        lenti_DMS=config['lenti_dms_REGN10933']
    output:
        nb_markdown=nb_markdown('REGN10933_yeast_lenti_dms_comparison.ipynb')
    params:
        nb='REGN10933_yeast_lenti_dms_comparison.ipynb'
    shell:
        "python scripts/run_nb.py {params.nb} {output.nb_markdown}"

