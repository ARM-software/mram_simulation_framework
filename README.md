<!-- vim-markdown-toc GFM -->

* [Arm's MRAM Simulation/Characterization Framework](#arms-mram-simulationcharacterization-framework)
	* [Authors and Related Publications](#authors-and-related-publications)
	* [Quick Start & More info](#quick-start--more-info)
	* [Files organization](#files-organization)
	* [s-LLGS Solvers](#s-llgs-solvers)
		* [No-thermal or emulated-thermal simulations](#no-thermal-or-emulated-thermal-simulations)
		* [Thermal Stochastic Simulations](#thermal-stochastic-simulations)

<!-- vim-markdown-toc -->
# Arm's MRAM Simulation/Characterization Framework

## Authors and Related Publications

* Fernando Garcia Redondo, fernando.garciaredondo@arm.com
* Pranay Prabhat
* Mudit Bhargava

Thanks to Cyrille Dray and Milos Milosavljevic for his helpful discussions.

The following frameworks have been presented at
* ***A Compact Model for Scalable MTJ Simulation***, IEEE International Conference on Synthesis, Modeling, Analysis and Simulation Methods and Applications to Circuit Design, SMACD 2021. [link to manuscript](https://arxiv.org/abs/2106.04976)
* ***A Fokker-Planck Solver to Model MTJ Stochasticity*** European Solid-State Device Research Conference, ESSDERC 2021. [link to manuscript](https://arxiv.org/abs/2106.12304)

This repository contains a framework for the characterization
and analysis of MRAM behavior including stochasticity, 
and a compact model and framework
for the efficient and scalable simulation of circuits with MRAMs.

We provide Verilog-A and Python compact models, able to emulate the behavior of
MRAMs switching at significant statistic events.
To calibrate the models for such stochastic based events, 
we implemented and analyzed two FPE solvers (numerical FVM and analytical), and
presented an optimization module that orchestrates the efficient computation 
of MRAM statistics and parameter regression.

## Quick Start & More info
Summary:
* `test_sllgs_solver.py` shows you the basic s-LLGS solver config, calling (`sllgs_solver.py`)
* `stochastic_multithread_simulation.py` (calling `sllgs_solver.py`) is the script
that helps you launching parallel python s-LLGS simulations
* These results can be compared against Fooker-Plank simulations (see `plot_sllgs_fpe_comparison.py` script)
* `analytical.py` and `mtj_fp_fvm.py` contain the Fooker-Plank solvers. Analytical contains the WER/RER fitter for the problem optimization
* Verilog-a compact models: run the testbenches `tb.scs` and `tb_subckt.scs`

Please, read the full description at [MRAM Framework Description](./doc/README.md).

**IMPORTANT: Before using the compact models**, read the [s-LLGS Solvers](#s-llgs-solvers) info.


## Files organization
* `doc`
	* [README.md](./doc/README.md) for the **full MRAM framework description**
* `src`
	* `python_compact_model`
		* [README.md](./python_compact_model/README.md) for the MRAM python s-LLGS description
		* `sllgs_solver.py` Python s-LLGS solver
		* `stochastic_multithread_simulation.py` Multi-thread stochastic simulations
		* `test_sllgs_solver.py` Simple s-LLGS tests
		* `ode_solver_custom_fn.py` *solve_ivp* auxilar fns derived from Scipy
	* `sllgs_fpe_comparison`
		* `plot_sllgs_fpe_comparison.py` Script for s-LLGS/Fooker-Plank comparison
		* `sllgs_importer.py` Script for importing `stochastic_multithread_simulation.py` results
	* `fokker_plank`
		* [README.md](./fokker_plank/README.md) for the MRAM Fokker-Plank description
		* `fvm`
			* `fvm_classes.py` Finite Volume Method classes, see [FVM](https://github.com/danieljfarrell/FVM)
			* `mtj_fp_fvm.py` MTJ Fokker-Plank FVM solver
		* `analytical`
			* `analytical.py` MTJ Fokker-Plank Analytical solver
			and WER/RER curves fitter
	* `verilog_a_compact_model`
		* [README.md](./verilog_a_compact_model/README.md) for the MRAM verilog-a compact model description
		* `tb` Testbenches
			* `tb.scs` Example testbench calling full Verilog-a model (conduction and s-LLGS fully written in verilog-a)
			* `tb_subckt.scs` Example testbench calling full Spectre subcircuit model (s-LLGS fully written in verilog-a, conduction writen in Spectre)
		* `mram_lib` Verilog-a compact model and Spectre library
			* `llg_spherical_solver.va` Verilog-a s-LLGS solver, key file
			* `*.va` Parameters or auxiliar functions
			* `*.scs` Spectre subcircuits and library
	

## s-LLGS Solvers

### No-thermal or emulated-thermal simulations
* Use Scipy solver in python (`scipy_ivp=True`)
* Use Spherical coordinates
* Control the simulation through tolerances (`atol, rtol` in python)

```
    # No thermal, no fake_thermal, solved with scipy_ivp
    llg_a = sllg_solver.LLG(do_fake_thermal=False,
                            do_thermal=False,
                            i_amp_fn=current,
                            seed=seed_0)
    data_a = llg_a.solve_and_plot(15e-9,
                                  scipy_ivp=True,
                                  solve_spherical=True,
                                  solve_normalized=True,
                                  rtol=1e-4,
                                  atol=1e-9)
    # No thermal, EMULATED THERMAL, solved with scipy_ivp
    llg_b = sllg_solver.LLG(do_fake_thermal=True,
                            d_theta_fake_th=1/30,
                            do_thermal=False,
                            i_amp_fn=current,
                            seed=seed_0)
    data_b = llg_b.solve_and_plot(15e-9,
                                  scipy_ivp=True,
                                  solve_spherical=True,
                                  solve_normalized=True,
                                  max_step=1e-11,
                                  rtol=1e-4,
                                  atol=1e-9)
```
### Thermal Stochastic Simulations
Require stochastic differential equation solvers
* Use SDE solvers in python (`scipy_ivp=False`)
* Use Cartesian coordinates
* Control the simulation through maximum time step (`max_step` in python)
```
    llg_c = sllg_solver.LLG(do_fake_thermal=False,
                            do_thermal=True,
                            i_amp_fn=current,
                            seed=seed_0)
    data_c = llg_c.solve_and_plot(10e-9,
                                  scipy_ivp=False,
                                  solve_spherical=False,
                                  solve_normalized=True,
                                  max_step=1e-13,
                                  method='stratonovich_heun')
```
![MRAM Magnetization and stochasticity](./doc/fig4_movie.gif)
