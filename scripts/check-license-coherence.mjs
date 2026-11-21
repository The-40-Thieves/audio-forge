#!/usr/bin/env node
/**
 * License & NOTICE coherence check for Apache-2.0 repositories (THE-1711).
 *
 * Verifies:
 *   1. Root `LICENSE` is Apache License, Version 2.0.
 *   2. Root `NOTICE` exists, is non-empty, and names the project and org.
 *   3. Root `Cargo.toml` declares `license = "Apache-2.0"` and every workspace
 *      member manifest agrees (`license.workspace = true` or `license = "Apache-2.0"`).
 *   4. `README.md` references Apache-2.0.
 *   5. `reaper-jsfx/*.jsfx` headers declare Apache-2.0 without conflicting MIT dual-license text.
 *   6. `.github/workflows/release.yml` packages `NOTICE` alongside `LICENSE`.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];

function check(condition, message) {
  if (!condition) {
    failures.push(message);
  }
}

function readUtf8(relPath) {
  const full = join(ROOT, relPath);
  if (!existsSync(full)) {
    failures.push(`Missing required file: ${relPath}`);
    return null;
  }
  return readFileSync(full, "utf8");
}

// 1. Root LICENSE must be Apache-2.0
const licenseText = readUtf8("LICENSE");
if (licenseText !== null) {
  check(
    licenseText.includes("Apache License") &&
      licenseText.includes("Version 2.0, January 2004"),
    "LICENSE does not appear to be Apache License, Version 2.0",
  );
}

// 2. Root NOTICE must exist and identify the project & org
const noticeText = readUtf8("NOTICE");
if (noticeText !== null) {
  check(noticeText.trim().length > 0, "NOTICE is empty");
  check(
    noticeText.includes("The 40 Thieves"),
    "NOTICE must mention 'The 40 Thieves'",
  );
  check(
    noticeText.includes("audio-forge"),
    "NOTICE must mention 'audio-forge'",
  );
}

// 3. Root Cargo.toml + every workspace member Cargo.toml
const rootCargo = readUtf8("Cargo.toml");
if (rootCargo !== null) {
  check(
    /^\s*license\s*=\s*"Apache-2\.0"\s*$/m.test(rootCargo),
    'Root Cargo.toml must declare `license = "Apache-2.0"`',
  );

  const membersMatch = rootCargo.match(/members\s*=\s*\[([\s\S]*?)\]/);
  if (membersMatch) {
    const members = [...membersMatch[1].matchAll(/"([^"]+)"/g)].map(
      (m) => m[1],
    );
    for (const member of members) {
      const rel = join(member, "Cargo.toml");
      const text = readUtf8(rel);
      if (text !== null) {
        const ok =
          /^\s*license\.workspace\s*=\s*true\s*$/m.test(text) ||
          /^\s*license\s*=\s*"Apache-2\.0"\s*$/m.test(text);
        check(
          ok,
          `${rel} must declare \`license.workspace = true\` or \`license = "Apache-2.0"\``,
        );
      }
    }
  } else {
    failures.push("Could not parse [workspace] members in Cargo.toml");
  }
}

// 4. README.md must reference Apache-2.0
const readme = readUtf8("README.md");
if (readme !== null) {
  check(
    readme.includes("Apache-2.0") || readme.includes("Apache License, Version 2.0"),
    "README.md must reference Apache-2.0",
  );
}

// 5. REAPER JSFX headers must agree on Apache-2.0
const jsfxDir = join(ROOT, "reaper-jsfx");
if (existsSync(jsfxDir)) {
  for (const name of readdirSync(jsfxDir).filter((f) => f.endsWith(".jsfx"))) {
    const rel = join("reaper-jsfx", name);
    const text = readUtf8(rel);
    if (text !== null) {
      check(
        text.includes("Apache-2.0") && !text.includes("MIT / Apache-2.0"),
        `${rel} header must declare Apache-2.0 (not stale 'MIT / Apache-2.0')`,
      );
    }
  }
}

// 6. Release workflow must package NOTICE alongside LICENSE (Apache-2.0 §4(d))
const releaseYml = readUtf8(".github/workflows/release.yml");
if (releaseYml !== null) {
  check(
    releaseYml.includes("LICENSE") && releaseYml.includes("NOTICE"),
    ".github/workflows/release.yml must package NOTICE alongside LICENSE",
  );
}

if (failures.length > 0) {
  console.error("License coherence check FAILED:");
  for (const f of failures) {
    console.error(`  - ${f}`);
  }
  process.exit(1);
}

console.log("License & NOTICE coherence check passed (Apache-2.0).");
