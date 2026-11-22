#![allow(semicolon_in_expressions_from_non_local_macros)]
use forge_plug::ForgeVocalTunePlug;
use nih_plug::prelude::*;

nih_export_clap!(ForgeVocalTunePlug);
nih_export_vst3!(ForgeVocalTunePlug);
