#!/usr/bin/env node
/**
 * License & NOTICE coherence check for Apache-2.0 repositories (THE-1711).
 *
 * Verifies:
 *   1. Root `LICENSE` is Apache License, Version 2.0.
 *   2. Root `NOTICE` exists, is non-empty, and names the project and org.
 *   3. Root `Cargo.toml` `[workspace.package]` declares `license = "Apache-2.0"` and every
 *      workspace member `[package]` agrees (`license.workspace = true` or `license = "Apache-2.0"`).
 *   4. `python-worker/pyproject.toml` `[project]` declares `Apache-2.0`.
 *   5. `README.md` references Apache-2.0.
 *   6. `reaper-jsfx/*.jsfx` exists, is non-empty, and every header declares `Apache-2.0`
 *      without conflicting dual-license text.
 *   7. `.github/workflows/release.yml` copies `NOTICE` into the release archive alongside `LICENSE`.
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

function tomlSection(text, header) {
  const escaped = header.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`^\\[${escaped}\\]\\s*$`, "m");
  const match = re.exec(text);
  if (!match) {
    return null;
  }
  const rest = text.slice(match.index + match[0].length);
  const nextHeader = /^\[[^\]]+\]\s*$/m.exec(rest);
  return nextHeader ? rest.slice(0, nextHeader.index) : rest;
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

// 3. Root Cargo.toml [workspace.package] + every workspace member [package]
const rootCargo = readUtf8("Cargo.toml");
if (rootCargo !== null) {
  const wsPkg = tomlSection(rootCargo, "workspace.package");
  check(
    wsPkg !== null && /^\s*license\s*=\s*"Apache-2\.0"\s*$/m.test(wsPkg),
    'Root Cargo.toml [workspace.package] must declare `license = "Apache-2.0"`',
  );

  const wsSection = tomlSection(rootCargo, "workspace");
  const membersMatch = wsSection?.match(/members\s*=\s*\[([\s\S]*?)\]/);
  if (membersMatch) {
    const members = [...membersMatch[1].matchAll(/"([^"]+)"/g)].map(
      (m) => m[1],
    );
    check(members.length > 0, "Cargo.toml [workspace].members must not be empty");
    for (const member of members) {
      const rel = join(member, "Cargo.toml");
      const text = readUtf8(rel);
      if (text !== null) {
        const pkg = tomlSection(text, "package");
        const ok =
          pkg !== null &&
          (/^\s*license\.workspace\s*=\s*true\s*$/m.test(pkg) ||
            /^\s*license\s*=\s*"Apache-2\.0"\s*$/m.test(pkg));
        check(
          ok,
          `${rel} [package] must declare \`license.workspace = true\` or \`license = "Apache-2.0"\``,
        );
      }
    }
  } else {
    failures.push("Could not parse [workspace] members in Cargo.toml");
  }
}

// 4. python-worker/pyproject.toml [project] must declare Apache-2.0
const pyproject = readUtf8("python-worker/pyproject.toml");
if (pyproject !== null) {
  const project = tomlSection(pyproject, "project");
  const pyOk =
    project !== null &&
    (/^\s*license\s*=\s*"Apache-2\.0"\s*$/m.test(project) ||
      /^\s*license\s*=\s*\{\s*text\s*=\s*"Apache-2\.0"\s*\}\s*$/m.test(project));
  check(
    pyOk,
    'python-worker/pyproject.toml [project] must declare Apache-2.0 license',
  );
}

// 5. README.md must reference Apache-2.0
const readme = readUtf8("README.md");
if (readme !== null) {
  check(
    readme.includes("Apache-2.0") ||
      readme.includes("Apache License, Version 2.0"),
    "README.md must reference Apache-2.0",
  );
}

// 6. REAPER JSFX headers must exist and agree strictly on Apache-2.0
const jsfxDir = join(ROOT, "reaper-jsfx");
if (!existsSync(jsfxDir)) {
  failures.push("Missing required directory: reaper-jsfx");
} else {
  const jsfxFiles = readdirSync(jsfxDir).filter((f) => f.endsWith(".jsfx"));
  check(jsfxFiles.length > 0, "reaper-jsfx/ must contain at least one .jsfx file");
  for (const name of jsfxFiles) {
    const rel = join("reaper-jsfx", name);
    const text = readUtf8(rel);
    if (text !== null) {
      const header = text.split(/\r?\n/).slice(0, 15).join("\n");
      check(
        /^\s*\/\/author:.*\(Apache-2\.0\)\s*$/m.test(header) &&
          !/\bMIT\b/i.test(header),
        `${rel} header must declare '(Apache-2.0)' without conflicting MIT license text`,
      );
    }
  }
}

// 7. Release workflow must explicitly copy NOTICE into dist/$out alongside LICENSE
const releaseYml = readUtf8(".github/workflows/release.yml");
if (releaseYml !== null) {
  const copiesLicense = /^\s*cp\b.*\bLICENSE\b.*dist\/\$out\//m.test(releaseYml);
  const copiesNotice = /^\s*(?:if\s+\[\s+-f\s+NOTICE\s+\];\s*then\s+)?cp\b.*\bNOTICE\b.*dist\/\$out\//m.test(
    releaseYml,
  );
  check(
    copiesLicense && copiesNotice,
    ".github/workflows/release.yml must copy both LICENSE and NOTICE into dist/$out/",
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
