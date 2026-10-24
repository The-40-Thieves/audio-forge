#![allow(semicolon_in_expressions_from_non_local_macros)]
use forge_plug::ForgeOpenSoothePlug;
use nih_plug::prelude::*;

nih_export_clap!(ForgeOpenSoothePlug);
nih_export_vst3!(ForgeOpenSoothePlug);
