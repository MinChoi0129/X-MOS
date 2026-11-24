#!/bin/bash
# Patches the sources of MinkowskiEngine 0.5.4 in the current directory for CUDA 12 / PyTorch 2.
set -e

# Thrust does not include these headers transitively anymore
sed -i '1i #include <thrust/execution_policy.h>' src/convolution_kernel.cuh src/3rdparty/concurrent_unordered_map.cuh
sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu
sed -i '1i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu

# The bundled copy of nvtx3.hpp clashes with the nvtx3 namespace of the CUDA toolkit
grep -rlE "namespace nvtx3|nvtx3::" src | xargs sed -i -E 's/namespace nvtx3/namespace me_nvtx3/g; s/nvtx3::/me_nvtx3::/g'

# nvcc can not resolve std::__to_address when a shared_ptr is assigned from a unique_ptr with a
# custom deleter, so hand over the raw pointer and the deleter explicitly
python3 - <<'PY'
path = "src/coordinate_map_gpu.cuh"
source = open(path).read()
old = """      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
"""
new = """      auto unique_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      auto map_deleter = unique_map.get_deleter();
      m_map = std::shared_ptr<map_type>(unique_map.release(), map_deleter);
"""
assert old in source
open(path, "w").write(source.replace(old, new))
PY
