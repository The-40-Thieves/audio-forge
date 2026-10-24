#![allow(semicolon_in_expressions_from_non_local_macros)]
use forge_plug::ForgeSubAndTapePlug;
use nih_plug::prelude::*;

nih_export_clap!(ForgeSubAndTapePlug);
nih_export_vst3!(ForgeSubAndTapePlug);
