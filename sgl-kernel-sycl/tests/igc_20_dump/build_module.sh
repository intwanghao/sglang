#!/usr/bin/env bash
rm -f module.o
rm -f module.pisa*
igc_lib="${IGC20_ROOT}/lib"
LD_LIBRARY_PATH=$IGC20_ROOT/lib:$LD_LIBRARY_PATH ocloc compile -device jgs -spirv_input -gen_file -output_no_suffix -internal_options -emit-pisa -file module.spv 2>&1

rm -f module.bin
mv module.gen module.pisa
# full options:  -ze-intel-enable-auto-large-GRF-mode  -ze-opt-level=2
$IGC20_ROOT/bin/llc -march=xe -swsb-allocation     module.pisa 
$IGC20_ROOT/bin/llc -march=xe -swsb-allocation -filetype=obj     module.pisa
