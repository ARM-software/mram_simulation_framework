## Quick Start & More info
Summary:
* Fokker-Plank simulations (see `plot_sllgs_fpe_comparison.py` script) `analytical.py` and `mtj_fp_fvm.py`

Please, read the full description at [MRAM Framework Description](./doc/README.md).


## Files organization
* `doc`
	* [README.md](./doc/README.md) for the **full MRAM framework description**
* `src`
	* `fokker_plank`
		* [README.md](./fokker_plank/README.md) for the MRAM Fokker-Plank description
		* `fvm`
			* `fvm_classes.py` Finite Volume Method classes, see [FVM](https://github.com/danieljfarrell/FVM)
			* `mtj_fp_fvm.py` MTJ Fokker-Plank FVM solver
		* `analytical`
			* `analytical.py` MTJ Fokker-Plank Analytical solver
			and WER/RER curves fitter

