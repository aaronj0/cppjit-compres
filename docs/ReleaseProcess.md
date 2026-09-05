# Release Process

How cppjit is versioned, tagged, and published. PyPI is the only
distribution channel today; conda and other channels get their own
sections when they exist. The mechanics follow the converged practice
of the large scientific-python projects: the test suite runs inside the
wheel build, PyPI uploads use trusted publishing, the README is the
project description, and the changelog lives on the GitHub releases
page.

## Where the release metadata lives

- `python/cppjit/_version.py` — the single version source (PEP 440).
  scikit-build-core reads it at build time through the
  `[[tool.dynamic-metadata]]` regex provider in `pyproject.toml`.
- `pyproject.toml` — everything PyPI displays: name, description, the
  README as the long description, license, classifiers, and URLs. PyPI
  takes these from the metadata inside the uploaded sdist and wheels;
  nothing is configured on pypi.org beyond the trusted publisher.
- `docs/ReleaseNotes.md` — the rolling release notes (next section).

## Release notes

`docs/ReleaseNotes.md` describes the release under development, in the
style of LLVM and CppInterOp: the release PR finalizes it, and the
post-release PR resets it for the next version. The notes of a released
version are preserved as its GitHub release body.

PyPI itself has no release-notes field. Each uploaded version renders
the README that was built into its metadata as the project page, so
notes reach PyPI users through the `Changelog` entry in
`[project.urls]`, which points at the releases page.

## The release ritual

1. Open the release PR: set `__version__` in
   `python/cppjit/_version.py` to the release version and finalize
   `docs/ReleaseNotes.md`. The `build-wheels` label gives the PR a full
   wheel build as pre-merge validation.
2. Merge the PR and tag the merge commit (`vX.Y.Z`). The tag push
   builds the full python list on every platform.
3. When that Wheels run finishes green, the PyPI workflow fires on its
   completion and publishes its wheels and sdist through trusted
   publishing; the `pypi` environment carries any reviewer approval
   configured for it. A red run publishes nothing: fix and re-tag.
4. Create the GitHub release for the tag: the body is this release's
   section of `docs/ReleaseNotes.md`, and the attached artifacts are
   the published run's wheels and sdist.
5. Open the post-release PR: set `_version.py` to the next planned
   version with a `.dev0` suffix (`0.1.0a2.dev0` after `0.1.0a1`) and
   reset `docs/ReleaseNotes.md` for that version.

The manual fallback is `workflow_dispatch` on the PyPI workflow with
the run id of a green full Wheels run. The workflow reacts to Wheels
runs from its copy on main, so it must have landed before the first
tag.

## PyPI publishing

The PyPI workflow (`.github/workflows/pypi.yml`) publishes the
artifacts of one finished Wheels run, so what lands on PyPI is byte
identical to what CI built and tested; nothing is rebuilt at publish
time. The workflow is registered as the project's trusted publisher on
pypi.org: the job mints a short-lived OIDC token, and no credential is
stored in the repository.

A version can be uploaded exactly once. A broken upload cannot be
replaced under the same version, even after deletion: delete it on PyPI
and release the next patch or prerelease number instead.

## Conda

Not yet distributed. This section gains the feedstock process when a
conda-forge recipe exists.
