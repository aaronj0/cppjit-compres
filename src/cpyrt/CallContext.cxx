// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "CallContext.h"

//- data _____________________________________________________________________
namespace cppjit::cpyrt {

CallContext::ECallFlags CallContext::sMemoryPolicy = CallContext::kUseStrict;
// this is just a data holder for linking; actual value is set in
// cpyrtModule.cxx
CallContext::ECallFlags CallContext::sSignalPolicy = CallContext::kNone;

} // namespace cppjit::cpyrt

//-----------------------------------------------------------------------------
void cpyrt::CallContext::AddTemporary(PyObject* pyobj) {
  if (pyobj) {
    if (!fTemps)
      fTemps = new Temporary{pyobj, nullptr};
    else {
      Temporary* tmp = fTemps;
      while (tmp->fNext)
        tmp = tmp->fNext;
      tmp->fNext = new Temporary{pyobj, nullptr};
    }
  }
}

//-----------------------------------------------------------------------------
void cpyrt::CallContext::Cleanup() {
  Temporary* tmp = fTemps;
  while (tmp) {
    Py_DECREF(tmp->fPyObject);
    Temporary* tmp2 = tmp->fNext;
    delete tmp;
    tmp = tmp2;
  }
  fTemps = nullptr;
}

//-----------------------------------------------------------------------------
bool cpyrt::CallContext::SetMemoryPolicy(ECallFlags e) {
  // Set the global memory policy, which affects object ownership when objects
  // are passed as function arguments.
  if (kUseHeuristics == e || e == kUseStrict) {
    sMemoryPolicy = e;
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
bool cpyrt::CallContext::SetGlobalSignalPolicy(bool setProtected) {
  // Set the global signal policy: under kProtected, method calls run inside
  // CppInterOp's signal guard and a fatal signal raised in C++ comes back as
  // a Python exception instead of terminating the process. The guard's
  // handlers are installed and removed together with the policy.
  bool old = sSignalPolicy == kProtected;
  sSignalPolicy = setProtected ? kProtected : kNone;
  Cpp::EnableSignalProtection(setProtected);
  return old;
}
