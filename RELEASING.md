# Releasing audio-forge

The org convention (see the organization's CONTRIBUTING): semver tags `vX.Y.Z`, one GitHub Release per tag, and the Release body is the changelog excerpt for that version. The product page at <https://the40thieves.dev/audio-forge/> reads the latest GitHub Release, so a tag without a Release does not appear there.

## Steps

1. Bump `version` under `[workspace.package]` in `Cargo.toml` and write the release notes for the version. Land that through a pull request like any other change.
2. Tag the merge commit on `main` and push the tag:

    ```bash
    git tag -s vX.Y.Z -m "audio-forge vX.Y.Z"
    git push origin vX.Y.Z
    ```

3. `release.yml` runs on the tag, builds the release artifacts and creates the GitHub Release. It can also be re-run by hand for an existing tag (workflow dispatch).
4. Edit the Release body so it is the release notes for this version, nothing else. The hub shows the first paragraph of it. Until the CI hardening work lands, note in the body that the artifacts are unsigned.

## What the hub reads

`hub.json` at the repository root (tagline, one measured claim with its source and date, install snippets per client, docs path). The install snippets build from source until a package or binary release exists; replace them when that changes. Update the claim's `measured_on` date whenever the conformance tests are re-run; the hub build rejects placeholder copy.
