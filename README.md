QEM with LSTM/ANN

For Vincenzo:

where is zne: zne is in zne/zne.ipynb and zne/zne_mitiq.ipynb

their data: in the data folder it's the 2 files called ising_init_from_qasm_coherent and no_readout

my data is in: circuits - for qc objects and features - that is already the pytorch conversion of the qc into lists and whatnot

ising experiments:
ising_experiment_our_model.ipynb - the one I've showed you
ising_experiment_our_model1 - 2.ipynb with increased hidden_size param
ising_experiment_rf.ipynb - their code

random circ experiments:
random_circs_paper_experiment.ipynb - experiments 1 - 18 circ depth
random_circs_paper_experiment_18_30.ipynb - experiments 18 - 30 circ depth
random_circs_paper_experiment_30_40.ipynb - experiments 30 - 40 circ depth
data generation for this is in: random_circs_data_gen.ipynb

test1 and test2 are just files where I was checking whether the code is doing what it's supposed to (plust checking whether accum grad helps or not)

the model scripts are in scripts.model

If there's anything not working with requirements or anything else let me know that I've missed, lmk. I will finish the other experiments we talked about.
