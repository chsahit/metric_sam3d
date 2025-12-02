notes on install
- follow sam3d-objects docs
- follow scenecomplete docs
- run setup_envs_properly.sh


● Here's a summary of the changes I made and which ones are actually necessary:

  Changes Made

  1. build_all_conda.sh (Line 12)

  Change: Added --no-build-isolation flag
  # Before:
  python -m pip install -e .

  # After:
  python -m pip install --no-build-isolation -e .
  Necessity: REQUIRED - Modern pip uses isolated build environments by default, which don't have access to torch in your conda environment.

  2. bundlesdf/mycuda/setup.py (Line 18)

  Change: Added --allow-unsupported-compiler to nvcc_flags
  # Before:
  nvcc_flags = ['-Xcompiler', '-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']

  # After:
  nvcc_flags = ['-Xcompiler', '-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', '--allow-unsupported-compiler']
  Necessity: OPTIONAL (probably) - This was added as a workaround, but the real issue was environment contamination. With a clean environment, GCC 11 should work without this flag. However, keeping it
  doesn't hurt and provides a safety net.

  3. Environment Fix (Not a file change)

  The critical fix: Unsetting NVCC_PREPEND_FLAGS before building
  unset NVCC_PREPEND_FLAGS
  Necessity: REQUIRED if you've previously activated other conda environments in the same terminal session.

  Recommended Approach Going Forward

  After following the README up to the build step, make these changes:

  Minimal necessary change (conservative approach):
  1. Modify build_all_conda.sh line 12 to add --no-build-isolation
  2. Run the build in a fresh terminal OR unset NVCC_PREPEND_FLAGS first

  Full approach (what we did - safest):
  1. Modify build_all_conda.sh to add --no-build-isolation
  2. Modify bundlesdf/mycuda/setup.py to add --allow-unsupported-compiler
  3. Unset NVCC_PREPEND_FLAGS before building

  Would you like me to test whether the --allow-unsupported-compiler flag is actually necessary by reverting it and trying a clean build?


- Install gdown in foundationpose env!
- modify download sxcript to skip all the inpainting weights
