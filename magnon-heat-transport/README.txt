This folder is for storage of data set for publication Phys. Rev. B 105, L161103, 
and follow up updata (publication version of data set is in zenodo record https://zenodo.org/record/6012346).




folder calculation/ is the data analysis folder.

alwaysBT_transport-annealing_single-U8.ipynb calculates the thermal conductivities for U=8 on the 8*8 lattice.
alwaysBT_transport-annealing_single-U10.ipynb calculates the thermal conductivities for U=10 on the 8*8 lattice.
alwaysBT_transport-annealing_single-U12.ipynb calculates the thermal conductivities for U=12 on the 8*8 lattice.
alwaysBT_transport-annealing_single-U12-dt.ipynb calculates the thermal conductivities for U=12 on the 8*8 lattice but with dtau=0.025 (others have dtau=0.05)
alwaysBT_transport-annealing_single-U{x}-{yy}.ipynb calculates the thermal conductivities for U=x on the y*y lattice.
transport-allflat-U8tp-0.25.ipynb calculates the thermal conductivities for U=8 on the 8*8 lattice with t'=-0.25t.

thermaldynamics.ipynb calculates the specific heat and compressibilities.
thermaldynamics-tp.ipynb calculates the specific heat and compressibilities for t'=-0.25t.
magnon_thermaldynamic.ipynb is for calculation according to the spin wave theory.
magnon_thermalconductivity.ipynb is for integrating the Drude weight of the kinetic part of the thermal condctivity.


zz_realspace-U{x}-{yy}.ipynb calculates the spin-spin correlation function on real space.
correlation_length.ipynb calcualtes the correlation length.

folder high-temperature-expansion/ is for the high temperature model function calculation.
main_tp0_all.cpp is the code to calculate the momentums.
a is the executable document.
go8_n1.sh is the script to run a.
U10tp0_8_n1 is the output for U=10,tp=0,n=1,highest order=8. Other output files are similar. Use "cat U10tp0_8_n1" to read the output.


folder plot/ is the folder for the plotting.


July 2nd, 2022 update:
minor improvement in raw data: 
Several contaminated Markov chains are targetted and removed in the analysis processes.
Contamination is due to unexpected interruption of writing measurements into h5 files.
No qualitative changes are caused in the plots and the conclusions in Phys. Rev. B 105, L161103 is not affected.
