# SARS-CoV-2 deep mutational scanning Delta library validations
In this project we validate SARS-CoV-2 Delta full spike deep muttaional scanning for two Delta breakthrou sera. 

Original Delta full spike deep mutational scanning data can be found [here](https://dms-vep.github.io/SARS-CoV-2_Delta_spike_DMS_REGN10933/).

The experimental steps are as follows:
- test Delta variant infectivity. Notebook: [virus_titers_Delta_serum_validation_mutants.ipynb](virus_titers_Delta_serum_validation_mutants.ipynb)
- perform virus neutralization experiments with sera. Notebook: [spike_neutralization-Delta_sera.ipynb](spike_neutralization-Delta_sera.ipynb)
- Compare REGN10933 DMS results between lentivirus and yeast-based DMS systems. Notebook: [REGN10933_yeast_lenti_dms_comparison.ipynb](REGN10933_yeast_lenti_dms_comparison.ipynb)

To run the analysis using snakemake pipeline type:

```
sbatch run_Hutch_cluster.bash
```

After the run is finished analysis results are found in [results](./results) folder.
