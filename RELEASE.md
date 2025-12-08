# Release Guide

This guide explains how to update and release new versions of `lumin-matrix` to PyPI.

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
  - **MAJOR**: Breaking changes
  - **MINOR**: New features (backward compatible)
  - **PATCH**: Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.1` → `0.2.0`: New feature
- `0.2.0` → `1.0.0`: Breaking change

## Release Workflow

### 1. Update Version Number

Update the version in **both** files:

**`pyproject.toml`:**
```toml
[project]
name = "lumin-matrix"
version = "0.1.2"  # ← Update this
```

**`setup.py`:**
```python
setup(
    name="lumin-matrix",
    version="0.1.2",  # ← Update this
    ...
)
```

### 2. Update CHANGELOG.md

Add an entry for the new version:
```markdown
## [0.1.2] - 2025-12-08

### Fixed
- Fixed issue with matrix multiplication on ARM64

### Changed
- Improved error messages
```

### 3. Test Locally

```bash
# Clean previous builds
rm -rf dist build _skbuild *.egg-info

# Build source distribution
python -m build --sdist

# Test installation from source
pip install dist/lumin_matrix-*.tar.gz

# Verify it works
python -c "import lumin; print(lumin.__version__)"
```

### 4. Build Distributions

```bash
# Build source distribution (required)
python -m build --sdist

# Optionally build wheel (if you have manylinux setup)
python -m build --wheel
```

### 5. Upload to PyPI

**Test on TestPyPI first (recommended):**
```bash
twine upload --repository testpypi dist/*
```

Then test installation:
```bash
pip install -i https://test.pypi.org/simple/ lumin-matrix
```

**Upload to real PyPI:**
```bash
twine upload dist/*
```

Or upload only source distribution:
```bash
twine upload dist/lumin_matrix-*.tar.gz
```

### 6. Create Git Tag (Optional but Recommended)

```bash
git tag -a v0.1.2 -m "Release version 0.1.2"
git push origin v0.1.2
```

## Quick Reference

```bash
# Full release workflow
vim pyproject.toml setup.py  # Update version
vim CHANGELOG.md             # Add changelog entry
rm -rf dist build _skbuild *.egg-info
python -m build --sdist
twine upload dist/lumin_matrix-*.tar.gz
git tag -a v0.1.2 -m "Release v0.1.2"
git push origin v0.1.2
```

## Troubleshooting

### "File already exists" error
- Version number already exists on PyPI
- Bump version number and try again

### Build fails
- Check that all dependencies are installed
- Verify CMakeLists.txt is correct
- Check that all source files are included in `sdist.include`

### Import errors after installation
- Verify the package structure is correct
- Check that `lumin/__init__.py` exists
- Ensure compiled module is in the right location

