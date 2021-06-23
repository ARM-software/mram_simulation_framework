## Quick Start & More info
Summary:
* Verilog-a compact models: run the testbenches `tb.scs` and `tb_subckt.scs`

Please, read the full description at [MRAM Framework Description](./doc/README.md).


## Files organization
* `doc`
	* [README.md](./doc/README.md) for the **full MRAM framework description**
* `src`
	* `verilog_a_compact_model`
		* [README.md](./verilog_a_compact_model/README.md) for the MRAM verilog-a compact model description
		* `tb` Testbenches
			* `tb.scs` Example testbench calling full Verilog-a model (conduction and s-LLGS fully written in verilog-a)
			* `tb_subckt.scs` Example testbench calling full Spectre subcircuit model (s-LLGS fully written in verilog-a, conduction writen in Spectre)
		* `mram_lib` Verilog-a compact model and Spectre library
			* `llg_spherical_solver.va` Verilog-a s-LLGS solver, key file
			* `*.va` Parameters or auxiliar functions
			* `*.scs` Spectre subcircuits and library
