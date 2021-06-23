# Arm's MRAM Simulation/Characterization Framework

## Quick Start & More info
Full readme: [README.md](../README.md)
Summary:
* `test_sllgs_solver.py` shows you the basic s-LLGS solver config, calling (`sllgs_solver.py`)
* `stochastic_multithread_simulation.py` (calling `sllgs_solver.py`) is the script
that helps you launching parallel python s-LLGS simulations
* These results can be compared against Fooker-Plank simulations (see `plot_sllgs_fpe_comparison.py` script)
* `analytical.py` and `mtj_fp_fvm.py` contain the Fooker-Plank solvers. Analytical contains the WER/RER fitter for the problem optimization
* Verilog-a compact models: run the testbenches `tb.scs` and `tb_subckt.scs`

Please, read the full description at [MRAM Framework Description](./doc/README.md).

