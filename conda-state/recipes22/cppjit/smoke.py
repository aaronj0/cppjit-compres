# Exercises the three conda-specific mechanisms: the bundled
# libclangCppInterOp loads, the bundled clang resource dir resolves builtin
# headers, and <vector> proves the run-dependency C++ standard headers.
import cppjit

cppjit.cppdef("""
#include <vector>
int conda_smoke(int x) { std::vector<int> v{x}; return v[0] + 1; }
""")
assert cppjit.gbl.conda_smoke(41) == 42

v = cppjit.gbl.std.vector["int"]()
v.push_back(7)
assert v[0] == 7
print("conda smoke passed")
