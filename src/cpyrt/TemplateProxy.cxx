// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "CPPClassMethod.h"
#include "CPPConstructor.h"
#include "CPPFunction.h"
#include "CPPInstance.h"
#include "CPPMethod.h"
#include "CPPOverload.h"
#include "LowLevelViews.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "TemplateProxy.h"
#include "Utility.h"
#include "cppjit_interop.h"

// Standard
#include <algorithm>

namespace cppjit::cpyrt {

static inline std::string targs2str(TemplateProxy* pytmpl) {
  if (!pytmpl || !pytmpl->fTemplateArgs)
    return "";
  return cpyrt_PyText_AsString(pytmpl->fTemplateArgs);
}

//----------------------------------------------------------------------------
TemplateInfo::TemplateInfo()
    : fPyClass(nullptr), fNonTemplated(nullptr), fTemplated(nullptr),
      fLowPriority(nullptr), fIsCUDAKernel(-1), fCUDALauncherName(nullptr),
      fDoc(nullptr) {
  /* empty */
}

//----------------------------------------------------------------------------
TemplateInfo::~TemplateInfo() {
  Py_XDECREF(fPyClass);

  Py_XDECREF(fCUDALauncherName);
  Py_XDECREF(fDoc);
  Py_DECREF(fNonTemplated);
  Py_DECREF(fTemplated);
  Py_DECREF(fLowPriority);

  for (const auto& p : fDispatchMap) {
    for (const auto& c : p.second) {
      Py_DECREF(c.second);
    }
  }
}

//----------------------------------------------------------------------------
void TemplateProxy::MergeOverload(CPPOverload* mp) {
  // Store overloads of this templated method.
  bool isGreedy = false;
  for (auto pc : mp->fMethodInfo->fMethods) {
    if (pc->IsGreedy()) {
      isGreedy = true;
      break;
    }
  }

  CPPOverload* cppol = isGreedy ? fTI->fLowPriority : fTI->fNonTemplated;
  cppol->MergeOverload(mp);
}

void TemplateProxy::AdoptMethod(PyCallable* pc) {
  // Store overload of this templated method.
  CPPOverload* cppol = pc->IsGreedy() ? fTI->fLowPriority : fTI->fNonTemplated;
  cppol->AdoptMethod(pc);
}

void TemplateProxy::AdoptTemplate(PyCallable* pc) {
  // Store known template methods.
  fTI->fTemplated->AdoptMethod(pc);
}

//----------------------------------------------------------------------------
PyObject* TemplateProxy::Instantiate(const std::string& fname,
                                     cpyrt_PyArgs_t args, size_t nargsf,
                                     Utility::ArgPreference pref, int* pcnt) {
  // Instantiate (and cache) templated methods, return method if any
  std::string proto = "";

  // adjust arguments for self if this is a rebound global function
  bool isNS = (((CPPScope*)fTI->fPyClass)->fFlags & CPPScope::kIsNamespace);
  if (!isNS && cpyrt_PyArgs_GET_SIZE(args, nargsf) &&
      (!fSelf || (fSelf == Py_None &&
                  !interop::IsStaticTemplate(
                      ((CPPScope*)fTI->fPyClass)->fCppType, fname)))) {
    args += 1;
    nargsf -= 1;
  }

  Py_ssize_t argc = cpyrt_PyArgs_GET_SIZE(args, nargsf);
  if (argc != 0) {
    PyObject* tpArgs = PyTuple_New(argc);
    for (Py_ssize_t i = 0; i < argc; ++i) {
      PyObject* itemi = cpyrt_PyArgs_GET_ITEM(args, i);

      bool bArgSet = false;

      // special case for arrays
      if (TemplateProxy_CheckExact(itemi)) {
        TemplateProxy* tn = (TemplateProxy*)itemi;
        PyObject* f = PyUnicode_FromFormat("%s%s", tn->fTI->fCppName.c_str(),
                                           targs2str(tn).c_str());
        PyTuple_SET_ITEM(tpArgs, i, f);
        bArgSet = true;
      }
      PyObject* pytc = nullptr;
      if (!bArgSet && (pytc = PyObject_GetAttr(itemi, PyStrings::gTypeCode))) {
        Py_buffer bufinfo;
        memset(&bufinfo, 0, sizeof(Py_buffer));
        std::string ptrdef;
        if (PyObject_GetBuffer(itemi, &bufinfo, PyBUF_FORMAT) == 0) {
          for (int j = 0; j < bufinfo.ndim; ++j)
            ptrdef += "*";
          PyBuffer_Release(&bufinfo);
        } else {
          ptrdef += "*";
          PyErr_Clear();
        }

        PyObject* pyptrname = Utility::CT2CppName(pytc, ptrdef.c_str(), true);
        if (pyptrname) {
          PyTuple_SET_ITEM(tpArgs, i, pyptrname);
          bArgSet = true;
          // string added, but not counted towards nStrings
        }
        Py_DECREF(pytc);
        pytc = nullptr;
      } else
        PyErr_Clear();

      // if not arg set, try special case for ctypes
      if (!bArgSet)
        pytc = PyObject_GetAttr(itemi, PyStrings::gCTypesType);

      if (!bArgSet && pytc) {
        PyObject* pyactname = Utility::CT2CppName(pytc, "&", false);
        if (!pyactname) {
          // _type_ of a pointer to c_type is that type, which will have a type
          PyObject* newpytc = PyObject_GetAttr(pytc, PyStrings::gCTypesType);
          Py_DECREF(pytc);
          pytc = newpytc;
          if (pytc) {
            pyactname = Utility::CT2CppName(pytc, "*", false);
          } else
            PyErr_Clear();
        }
        Py_XDECREF(pytc);
        pytc = nullptr;
        if (pyactname) {
          PyTuple_SET_ITEM(tpArgs, i, pyactname);
          bArgSet = true;
          // string added, but not counted towards nStrings
        }
      } else
        PyErr_Clear();

      if (!bArgSet) {
        // normal case (may well fail)
        PyErr_Clear();
        PyObject* tp = (PyObject*)Py_TYPE(itemi);
        Py_INCREF(tp);
        PyTuple_SET_ITEM(tpArgs, i, tp);
      }
    }

    PyObject* pyargs = PyTuple_New(argc);
    for (Py_ssize_t i = 0; i < argc; ++i) {
      PyObject* item = cpyrt_PyArgs_GET_ITEM(args, i);
      Py_INCREF(item);
      PyTuple_SET_ITEM(pyargs, i, item);
    }
    const std::string& name_v1 =
        Utility::ConstructTemplateArgs(nullptr, tpArgs, pyargs, pref, 0, pcnt);

    Py_DECREF(pyargs);
    Py_DECREF(tpArgs);

    // Propagate the error that occurs if we can't construct the C++ name
    // from the provided template argument
    if (PyErr_Occurred()) {
      return nullptr;
    }

    if (name_v1.size())
      proto = name_v1.substr(1, name_v1.size() - 2);
  }

  // the following causes instantiation as necessary
  interop::TCppScope_t scope = ((CPPClass*)fTI->fPyClass)->fCppType;
  interop::TCppMethod_t cppmeth =
      interop::GetMethodTemplate(scope, fname, proto);
  if (cppmeth) { // overload stops here
    // A successful instantiation needs to be cached to pre-empt future
    // instantiations. There are two names involved, the original asked (which
    // may be partial) and the received.
    //
    // Caching scheme: if the match is exact, simply add the overload to the
    // pre-existing one, or create a new overload for later lookups. If the
    // match is not exact, do the same, but also create an alias. Only add exact
    // matches to the set of known template instantiations, to prevent piling on
    // from different partial instantiations.
    //
    // TODO: this caches the lookup method before the call, meaning that failing
    // overloads can add already existing overloads to the set of methods.

    std::string resname =
        interop::GetFullName(interop::TCppScope_t(cppmeth.data));

    // An initializer_list is preferred for the argument types, but should not
    // leak into the argument types. If it did, replace with vector and lookup
    // anew.
    if (resname.find("initializer_list") != std::string::npos) {
      auto pos = proto.find("initializer_list");
      while (pos != std::string::npos) {
        proto.replace(pos, 16, "vector");
        pos = proto.find("initializer_list", pos + 6);
      }

      interop::TCppMethod_t m2 =
          interop::GetMethodTemplate(scope, fname, proto);
      if (m2 && m2 != cppmeth) {
        // replace if the new method with vector was found; otherwise just
        // continue with the previously found method with initializer_list.
        cppmeth = m2;
        resname = interop::GetFullName(interop::TCppScope_t(cppmeth.data));
      }
    }

    bool bExactMatch = fname == resname;

    // lookup on existing name in case this was an overload, not a caching,
    // failure
    PyObject* dct = PyObject_GetAttr(fTI->fPyClass, PyStrings::gDict);
    PyObject* pycachename = cpyrt_PyText_InternFromString(fname.c_str());
    PyObject* pyol = PyObject_GetItem(dct, pycachename);
    if (!pyol)
      PyErr_Clear();
    bool bIsCppOL = CPPOverload_Check(pyol);
    bool bIsCppTP = TemplateProxy_Check(pyol);

    // find the full name if the requested one was partial
    PyObject* exact = nullptr;
    PyObject* pyresname = cpyrt_PyText_FromString(resname.c_str());
    if (!bExactMatch) {
      exact = PyObject_GetItem(dct, pyresname);
      if (!exact)
        PyErr_Clear();
    }
    Py_DECREF(dct);

    bool bIsConstructor = false, bNeedsRebind = true;

    PyCallable* meth = nullptr;
    if (interop::IsNamespace(scope)) {
      meth = new CPPFunction(scope, cppmeth);
      bNeedsRebind = false;
    } else if (interop::IsStaticMethod(cppmeth)) {
      meth = new CPPClassMethod(scope, cppmeth);
      bNeedsRebind = false;
    } else if (interop::IsConstructor(cppmeth)) {
      bIsConstructor = true;
      meth = new CPPConstructor(scope, cppmeth);
    } else
      meth = new CPPMethod(scope, cppmeth);

    // Case 1/2: method simply did not exist before
    if (!pyol) {
      // actual overload to use (now owns meth)
      pyol = (PyObject*)CPPOverload_New(fname, meth);
      if (bIsConstructor) {
        // TODO: this is an ugly hack :(
        ((CPPOverload*)pyol)->fMethodInfo->fFlags |=
            CallContext::kIsCreator | CallContext::kIsConstructor;
      }

      // add to class dictionary
      PyType_Type.tp_setattro(fTI->fPyClass, pycachename, pyol);
    }

    // Case 3/4: pre-existing method that was either not found b/c the full
    // templated name was constructed in this call or it failed as overload
    else if (bIsCppOL) {
      // TODO: see above, since the call hasn't happened yet, this overload may
      // already exist and fail again.
      ((CPPOverload*)pyol)->AdoptMethod(meth); // takes ownership
    }

    // Case 5: must be a template proxy, meaning that current template name is
    // not a template overload
    else if (bIsCppTP) {
      ((TemplateProxy*)pyol)->AdoptTemplate(meth->Clone());
      Py_DECREF(pyol);
      pyol = (PyObject*)CPPOverload_New(fname, meth); // takes ownership
    }
    // Case 6: pre-existing object is not a CPPOverload nor TemplateProxy
    // we do not cache it, as this might be a pythonization (monkey-patched
    // func/method)
    else {
      Py_DECREF(pyol);
      pyol = (PyObject*)CPPOverload_New(fname, meth);
    }

    // Special Case if name was aliased (e.g. typedef in template instantiation)
    if (!exact && !bExactMatch) {
      PyType_Type.tp_setattro(fTI->fPyClass, pyresname, pyol);
    }

    // cleanup
    Py_XDECREF(exact);
    Py_DECREF(pyresname);
    Py_DECREF(pycachename);

    // retrieve fresh (for boundedness) and call
    PyObject* pymeth = CPPOverload_Type.tp_descr_get(
        pyol, bNeedsRebind ? fSelf : nullptr, (PyObject*)&CPPOverload_Type);
    Py_DECREF(pyol);
    return pymeth;
  }

  PyErr_Format(PyExc_TypeError, "Failed to instantiate \"%s(%s)\"",
               fname.c_str(), proto.c_str());
  return nullptr;
}

//= cpyrt template proxy construction/destruction =========================
static TemplateProxy* tpp_new(PyTypeObject*, PyObject*, PyObject*) {
  // Create a new empty template method proxy.
  TemplateProxy* pytmpl = PyObject_GC_New(TemplateProxy, &TemplateProxy_Type);
  pytmpl->fSelf = nullptr;
  pytmpl->fTemplateArgs = nullptr;
  pytmpl->fLaunchConfig = nullptr;
  pytmpl->fWeakrefList = nullptr;
  new (&pytmpl->fTI) TP_TInfo_t{};
  pytmpl->fTI = std::make_shared<TemplateInfo>();

  PyObject_GC_Track(pytmpl);
  return pytmpl;
}

//----------------------------------------------------------------------------
static Py_hash_t tpp_hash(TemplateProxy* self) { return (Py_hash_t)self; }

//----------------------------------------------------------------------------
static PyObject* tpp_richcompare(TemplateProxy* self, PyObject* other, int op) {
  if (op == Py_EQ || op == Py_NE) {
    if (!TemplateProxy_CheckExact(other))
      Py_RETURN_FALSE;

    if (self->fTI == ((TemplateProxy*)other)->fTI)
      Py_RETURN_TRUE;

    Py_RETURN_FALSE;
  }

  Py_INCREF(Py_NotImplemented);
  return Py_NotImplemented;
}

//----------------------------------------------------------------------------
static int tpp_clear(TemplateProxy* pytmpl) {
  // Garbage collector clear of held python member objects.
  Py_CLEAR(pytmpl->fSelf);
  Py_CLEAR(pytmpl->fTemplateArgs);
  Py_CLEAR(pytmpl->fLaunchConfig);

  return 0;
}

//----------------------------------------------------------------------------
static void tpp_dealloc(TemplateProxy* pytmpl) {
  // Destroy the given template method proxy.
  if (pytmpl->fWeakrefList)
    PyObject_ClearWeakRefs((PyObject*)pytmpl);
  PyObject_GC_UnTrack(pytmpl);
  tpp_clear(pytmpl);
  pytmpl->fTI.~TP_TInfo_t();
  PyObject_GC_Del(pytmpl);
}

//----------------------------------------------------------------------------
static int tpp_traverse(TemplateProxy* pytmpl, visitproc visit, void* arg) {
  // Garbage collector traverse of held python member objects.
  Py_VISIT(pytmpl->fSelf);
  Py_VISIT(pytmpl->fTemplateArgs);
  Py_VISIT(pytmpl->fLaunchConfig);

  return 0;
}

//----------------------------------------------------------------------------
static PyObject* tpp_doc(TemplateProxy* pytmpl, void*) {
  if (pytmpl->fTI->fDoc) {
    Py_INCREF(pytmpl->fTI->fDoc);
    return pytmpl->fTI->fDoc;
  }

  // Forward to method proxies to doc all overloads
  PyObject* doc = nullptr;
  if (pytmpl->fTI->fNonTemplated->HasMethods())
    doc = PyObject_GetAttrString((PyObject*)pytmpl->fTI->fNonTemplated,
                                 "__doc__");
  if (pytmpl->fTI->fTemplated->HasMethods()) {
    PyObject* doc2 =
        PyObject_GetAttrString((PyObject*)pytmpl->fTI->fTemplated, "__doc__");
    if (doc && doc2) {
      cpyrt_PyText_AppendAndDel(&doc, cpyrt_PyText_FromString("\n"));
      cpyrt_PyText_AppendAndDel(&doc, doc2);
    } else if (!doc && doc2) {
      doc = doc2;
    }
  }
  if (pytmpl->fTI->fLowPriority->HasMethods()) {
    PyObject* doc2 =
        PyObject_GetAttrString((PyObject*)pytmpl->fTI->fLowPriority, "__doc__");
    if (doc && doc2) {
      cpyrt_PyText_AppendAndDel(&doc, cpyrt_PyText_FromString("\n"));
      cpyrt_PyText_AppendAndDel(&doc, doc2);
    } else if (!doc && doc2) {
      doc = doc2;
    }
  }

  if (doc)
    return doc;

  return cpyrt_PyText_FromString(TemplateProxy_Type.tp_doc);
}

static int tpp_doc_set(TemplateProxy* pytmpl, PyObject* val, void*) {
  Py_XDECREF(pytmpl->fTI->fDoc);
  Py_INCREF(val);
  pytmpl->fTI->fDoc = val;
  return 0;
}

//----------------------------------------------------------------------------

//= cpyrt template proxy callable behavior ================================

#define TPPCALL_RETURN                                                         \
  {                                                                            \
    errors.clear();                                                            \
    return result;                                                             \
  }

static inline void UpdateDispatchMap(TemplateProxy* pytmpl, bool use_targs,
                                     uint64_t sighash, CPPOverload* pymeth) {
  // Memoize a method in the dispatch map after successful call; replace old if
  // need be (may be with the same CPPOverload, just with more methods).
  bool bInserted = false;
  auto& v = pytmpl->fTI->fDispatchMap[use_targs ? targs2str(pytmpl) : ""];

  Py_INCREF(pymeth);
  for (auto& p : v) {
    if (p.first == sighash) {
      Py_DECREF(p.second);
      p.second = pymeth;
      bInserted = true;
    }
  }
  if (!bInserted)
    v.push_back(std::make_pair(sighash, pymeth));
}

static inline PyObject*
SelectAndForward(TemplateProxy* pytmpl, CPPOverload* pymeth,
                 cpyrt_PyArgs_t args, size_t nargsf, PyObject* kwds,
                 bool implicitOkay, bool use_targs, uint64_t sighash,
                 std::vector<Utility::PyError_t>& errors) {
  // Forward a call to known overloads, if any.
  if (pymeth->HasMethods()) {
    PyObject* pycall = CPPOverload_Type.tp_descr_get(
        (PyObject*)pymeth, pytmpl->fSelf, (PyObject*)&CPPOverload_Type);

    if (!implicitOkay)
      ((CPPOverload*)pycall)->fFlags |= CallContext::kNoImplicit;

    // now call the method with the arguments (loops internally)
    PyObject* result = cpyrt_tp_call(pycall, args, nargsf, kwds);
    Py_DECREF(pycall);
    if (result) {
      UpdateDispatchMap(pytmpl, use_targs, sighash, pymeth);
      TPPCALL_RETURN;
    }
    Utility::FetchError(errors);
  }

  return nullptr;
}

static inline PyObject* CallMethodImp(TemplateProxy* pytmpl, PyObject*& pymeth,
                                      cpyrt_PyArgs_t args, size_t nargsf,
                                      PyObject* kwds, bool impOK,
                                      uint64_t sighash) {
  // Actual call of a given overload: takes care of handlign of "self" and
  // dereferences the overloaded method after use.

  PyObject* result;
  if (!impOK && CPPOverload_Check(pymeth))
    ((CPPOverload*)pymeth)->fFlags |= CallContext::kNoImplicit;
  bool isNS =
      (((CPPScope*)pytmpl->fTI->fPyClass)->fFlags & CPPScope::kIsNamespace);
  if (isNS && pytmpl->fSelf && pytmpl->fSelf != Py_None) {
    // this is a global method added a posteriori to the class
    PyCallArgs cargs{(CPPInstance*&)pytmpl->fSelf, args, nargsf, kwds};
    AdjustSelf(cargs);
    result = cpyrt_tp_call(pymeth, cargs.fArgs, cargs.fNArgsf, cargs.fKwds);
  } else {
    if (!pytmpl->fSelf && CPPOverload_Check(pymeth))
      ((CPPOverload*)pymeth)->fFlags &= ~CallContext::kFromDescr;
    result = cpyrt_tp_call(pymeth, args, nargsf, kwds);
  }

  if (result) {
    Py_XDECREF(((CPPOverload*)pymeth)->fSelf);
    ((CPPOverload*)pymeth)->fSelf = nullptr; // unbind
    UpdateDispatchMap(pytmpl, true, sighash, (CPPOverload*)pymeth);
  }

  Py_DECREF(pymeth);
  pymeth = nullptr;
  return result;
}

//----------------------------------------------------------------------------
static bool tpp_is_cuda_kernel(TemplateProxy* pytmpl) {
  // A proxy launches CUDA kernels iff its name resolves to at least one
  // __global__ function in its scope. Cached: a kernel definition cannot
  // be undone in the interpreter.
  TemplateInfo& ti = *pytmpl->fTI;
  if (ti.fIsCUDAKernel == -1) {
    ti.fIsCUDAKernel = 0;
    if (interop::IsCUDAEnabled() && CPPScope_Check(ti.fPyClass)) {
      interop::TCppScope_t scope = ((CPPScope*)ti.fPyClass)->fCppType;
      for (auto method : interop::GetMethodsFromName(scope, ti.fCppName))
        if (interop::IsCUDAFunction(method)) {
          ti.fIsCUDAKernel = 1;
          break;
        }
    }
  }
  return ti.fIsCUDAKernel == 1;
}

//----------------------------------------------------------------------------
static bool tpp_cuda_dims(PyObject* spec, unsigned long long dims[3],
                          const char* what) {
  // One grid/block spec: an int or a sequence of up to three ints.
  dims[0] = dims[1] = dims[2] = 1;
  Py_ssize_t n = 1;
  PyObject* seq = nullptr;
  if (!PyIndex_Check(spec)) {
    seq = PySequence_Fast(spec, "CUDA launch dimensions must be an int or "
                                "a sequence of up to three ints");
    if (!seq)
      return false;
    n = PySequence_Fast_GET_SIZE(seq);
    if (n < 1 || 3 < n) {
      Py_DECREF(seq);
      PyErr_Format(PyExc_ValueError, "%s takes one to three dimensions", what);
      return false;
    }
  }
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = seq ? PySequence_Fast_GET_ITEM(seq, i) : spec;
    unsigned long long v = PyLong_AsUnsignedLongLong(item);
    if (v == (unsigned long long)-1 && PyErr_Occurred()) {
      Py_XDECREF(seq);
      return false;
    }
    if (v == 0) {
      Py_XDECREF(seq);
      PyErr_Format(PyExc_ValueError, "%s dimensions must be positive", what);
      return false;
    }
    dims[i] = v;
  }
  Py_XDECREF(seq);
  return true;
}

//----------------------------------------------------------------------------
static bool tpp_cuda_stream_arg(PyObject* obj, unsigned long long& stream) {
  // The stream slot also takes objects speaking the __cuda_stream__
  // protocol (cuda.core, torch, cupy): (version, handle) with version 0,
  // provided as a method or as an already-built tuple attribute.
  PyObject* proto = PyObject_GetAttr(obj, PyStrings::gCudaStream);
  if (!proto) {
    PyErr_Clear();
    PyErr_SetString(PyExc_TypeError,
                    "the CUDA launch stream must be an int handle or "
                    "provide __cuda_stream__");
    return false;
  }
  PyObject* info = proto;
  if (!PyTuple_Check(proto)) {
    info = PyObject_CallObject(proto, nullptr);
    Py_DECREF(proto);
    if (!info)
      return false;
  }
  if (!PyTuple_Check(info) || PyTuple_GET_SIZE(info) != 2) {
    Py_DECREF(info);
    PyErr_SetString(PyExc_TypeError,
                    "__cuda_stream__ must provide (version, handle)");
    return false;
  }
  long version = PyLong_AsLong(PyTuple_GET_ITEM(info, 0));
  if (version != 0) {
    Py_DECREF(info);
    if (!PyErr_Occurred())
      PyErr_Format(PyExc_TypeError,
                   "unsupported __cuda_stream__ protocol version %ld", version);
    return false;
  }
  stream = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(info, 1));
  Py_DECREF(info);
  return !(stream == (unsigned long long)-1 && PyErr_Occurred());
}

//----------------------------------------------------------------------------
static PyObject* tpp_cuda_launch_config(PyObject* args) {
  // The subscript key of kern[grid, block(, shared_bytes(, stream))],
  // normalized to the launcher's eight leading scalars.
  PyObject* items[4];
  Py_ssize_t n = 1;
  if (PyTuple_Check(args)) {
    n = PyTuple_GET_SIZE(args);
    for (Py_ssize_t i = 0; i < n && i < 4; ++i)
      items[i] = PyTuple_GET_ITEM(args, i);
  } else {
    items[0] = args;
  }
  if (n < 2 || 4 < n) {
    PyErr_SetString(PyExc_TypeError, "CUDA kernels take a launch config: "
                                     "[grid, block(, shared_bytes(, stream))]");
    return nullptr;
  }

  unsigned long long dims[8] = {1, 1, 1, 1, 1, 1, 0, 0};
  if (!tpp_cuda_dims(items[0], dims, "grid") ||
      !tpp_cuda_dims(items[1], dims + 3, "block"))
    return nullptr;
  for (Py_ssize_t i = 2; i < n; ++i) {
    if (i == 3 && !PyIndex_Check(items[i])) {
      if (!tpp_cuda_stream_arg(items[i], dims[7]))
        return nullptr;
      continue;
    }
    dims[i + 4] = PyLong_AsUnsignedLongLong(items[i]);
    if (dims[i + 4] == (unsigned long long)-1 && PyErr_Occurred())
      return nullptr;
  }

  PyObject* config = PyTuple_New(8);
  for (int i = 0; i < 8; ++i)
    PyTuple_SET_ITEM(config, i, PyLong_FromUnsignedLongLong(dims[i]));
  return config;
}

//----------------------------------------------------------------------------
static bool tpp_cuda_same_stream(unsigned long long a, unsigned long long b) {
  // 0 and 1 both name the legacy default stream (2 is per-thread default).
  return a == b || ((a | b) == 1);
}

//----------------------------------------------------------------------------
static bool tpp_cuda_is_contiguous(PyObject* strides,
                                   const std::vector<dim_t>& shape,
                                   long itemsize) {
  // A kernel is handed a bare pointer and indexes it densely, so only a
  // C-contiguous buffer can be passed on: strides run itemsize,
  // itemsize*shape[n-1], ... from the last dimension backwards.
  if (!strides || strides == Py_None)
    return true; // the interfaces spell "dense" as no strides
  if (!PySequence_Check(strides) ||
      (size_t)PySequence_Size(strides) != shape.size())
    return false;
  dim_t expected = itemsize;
  for (Py_ssize_t i = (Py_ssize_t)shape.size() - 1; i >= 0; --i) {
    PyObject* s = PySequence_GetItem(strides, i);
    dim_t got = s ? PyLong_AsSsize_t(s) : -1;
    Py_XDECREF(s);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    if (got != expected)
      return false;
    expected *= shape[i];
  }
  return true;
}

//----------------------------------------------------------------------------
static int tpp_cuda_device_arg(PyObject* obj, unsigned long long stream,
                               PyObject*& out) {
  // Kernels take device memory. Buffers offered through the CUDA array
  // interface become typed views over their device pointer; host buffers
  // are refused rather than passed on as a pointer the device cannot
  // dereference. Returns 1 when `out` replaces the argument, 0 to keep
  // the argument as it is, -1 with an exception set.
  PyObject* cai = PyObject_GetAttr(obj, PyStrings::gCudaArrayInterface);
  if (!cai) {
    PyErr_Clear();
    // A DLPack exporter names the memory it holds: device memory needs the
    // capsule protocol that cppjit.cuda.view() implements, while host
    // memory is an error whichever protocol offers it.
    PyObject* dldev =
        PyObject_CallMethodObjArgs(obj, PyStrings::gDLPackDevice, nullptr);
    long device = -1;
    if (dldev) {
      if (PyTuple_Check(dldev) && PyTuple_GET_SIZE(dldev) == 2)
        device = PyLong_AsLong(PyTuple_GET_ITEM(dldev, 0));
      Py_DECREF(dldev);
    }
    PyErr_Clear();
    if (device == 2 || device == 13) { // kDLCUDA, kDLCUDAManaged
      PyErr_Format(PyExc_TypeError,
                   "'%s' offers its buffer through DLPack only; import it "
                   "with cppjit.cuda.view() and pass the view",
                   Py_TYPE(obj)->tp_name);
      return -1;
    }
    if (device != -1 || PyObject_CheckBuffer(obj) ||
        PyObject_HasAttr(obj, PyStrings::gArrayInterface)) {
      PyErr_Format(PyExc_TypeError,
                   "host memory ('%s') passed to a CUDA kernel",
                   Py_TYPE(obj)->tp_name);
      return -1;
    }
    return 0;
  }

  int status = -1;
  // PyDict_GetItemString requires an actual dict; every published exporter
  // provides one, but a mapping would otherwise take an unchecked path.
  if (!PyDict_Check(cai)) {
    PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__ must be a dict");
    Py_DECREF(cai);
    return -1;
  }
  PyObject* data = PyDict_GetItemString(cai, "data");
  PyObject* typestr = PyDict_GetItemString(cai, "typestr");
  PyObject* pyshape = PyDict_GetItemString(cai, "shape");
  PyObject* pyversion = PyDict_GetItemString(cai, "version");
  if (!data || !PyTuple_Check(data) || PyTuple_GET_SIZE(data) != 2 ||
      !typestr || !PyUnicode_Check(typestr) || !pyshape ||
      !PySequence_Check(pyshape)) {
    PyErr_SetString(PyExc_TypeError, "malformed __cuda_array_interface__");
    Py_DECREF(cai);
    return -1;
  }
  // Versions past 3 may add fields that change how the buffer is read.
  long version = pyversion ? PyLong_AsLong(pyversion) : 3;
  if (PyErr_Occurred() || version < 1 || version > 3) {
    PyErr_Clear();
    PyErr_Format(PyExc_TypeError,
                 "unsupported __cuda_array_interface__ version %ld", version);
    Py_DECREF(cai);
    return -1;
  }
  if (PyDict_GetItemString(cai, "mask") &&
      PyDict_GetItemString(cai, "mask") != Py_None) {
    PyErr_SetString(PyExc_TypeError, "masked arrays are not supported");
    Py_DECREF(cai);
    return -1;
  }

  unsigned long long ptr = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(data, 0));
  if (ptr != (unsigned long long)-1 || !PyErr_Occurred()) {
    std::vector<dim_t> dims;
    Py_ssize_t ndim = PySequence_Size(pyshape);
    for (Py_ssize_t i = 0; i < ndim; ++i) {
      PyObject* d = PySequence_GetItem(pyshape, i);
      dims.push_back(d ? PyLong_AsSsize_t(d) : -1);
      Py_XDECREF(d);
    }
    const char* ts =
        PyErr_Occurred() ? nullptr : cpyrt_PyText_AsString(typestr);
    long itemsize = ts ? strtol(ts + 2, nullptr, 10) : 0;
    if (!ts) {
      PyErr_Clear();
      PyErr_SetString(PyExc_TypeError, "malformed __cuda_array_interface__");
    } else if (!tpp_cuda_is_contiguous(PyDict_GetItemString(cai, "strides"),
                                       dims, itemsize)) {
      // The kernel gets a pointer, not a layout: a strided buffer would be
      // read as if it were dense, silently returning wrong results.
      PyErr_Format(PyExc_TypeError,
                   "'%s' is not C-contiguous; CUDA kernels take dense "
                   "buffers (copy it first)",
                   Py_TYPE(obj)->tp_name);
    } else {
      out = CreateLowLevelViewFromTypestr((void*)ptr, ts,
                                          dims_t(dims.size(), dims.data()));
      if (!out)
        PyErr_Format(PyExc_TypeError,
                     "no CUDA kernel argument type for buffers of type '%s'",
                     ts);
      else
        status = 1;
    }
  }

  // The producer names the stream its pending work runs on; ours must
  // wait for it before reading the buffer.
  PyObject* pystream =
      status == 1 ? PyDict_GetItemString(cai, "stream") : nullptr;
  if (pystream && pystream != Py_None) {
    unsigned long long producer = PyLong_AsUnsignedLongLong(pystream);
    if (producer == (unsigned long long)-1 && PyErr_Occurred()) {
      status = -1;
    } else if (producer == 0) {
      PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__ stream 0 is "
                                       "disallowed by the protocol");
      status = -1;
    } else if (!tpp_cuda_same_stream(producer, stream) &&
               !interop::CUDAStreamWait(producer, stream)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "could not order the launch after the buffer's stream");
      status = -1;
    }
    if (status < 0) {
      Py_CLEAR(out);
    }
  }

  Py_DECREF(cai);
  return status;
}

//----------------------------------------------------------------------------
static PyObject* tpp_cuda_launch(TemplateProxy* pytmpl, PyObject* const* args,
                                 size_t nargsf, PyObject* kwds) {
  // Dispatch to the runtime launcher AdaptCUDAFunction generated in the
  // kernel's scope, with the bound launch config prepended.
  if (kwds && PyTuple_GET_SIZE(kwds)) {
    PyErr_SetString(PyExc_TypeError,
                    "CUDA kernel launches take no keyword arguments");
    return nullptr;
  }
  TemplateInfo& ti = *pytmpl->fTI;
  if (!ti.fCUDALauncherName)
    ti.fCUDALauncherName = cpyrt_PyText_InternFromString(
        (interop::kCUDALaunchPrefix + ti.fCppName).c_str());

  PyObject* launcher = PyObject_GetAttr(ti.fPyClass, ti.fCUDALauncherName);
  if (!launcher)
    return nullptr;

  Py_ssize_t argc = cpyrt_PyArgs_GET_SIZE(args, nargsf);
  Py_ssize_t nconf = PyTuple_GET_SIZE(pytmpl->fLaunchConfig);
  std::vector<PyObject*> largs(nconf + argc);
  for (Py_ssize_t i = 0; i < nconf; ++i)
    largs[i] = PyTuple_GET_ITEM(pytmpl->fLaunchConfig, i);

  unsigned long long stream =
      nconf == 8 ? PyLong_AsUnsignedLongLong(
                       PyTuple_GET_ITEM(pytmpl->fLaunchConfig, 7))
                 : 0;
  std::vector<PyObject*> views; // substitutions, alive across the launch
  for (Py_ssize_t i = 0; i < argc; ++i) {
    PyObject* arg = args[i];
    largs[nconf + i] = arg;
    // the types the converters take as they are, checked before any
    // attribute lookup so ordinary launches pay nothing for this
    if (PyLong_CheckExact(arg) || PyFloat_CheckExact(arg) ||
        LowLevelView_Check(arg) || CPPInstance_Check(arg))
      continue;
    PyObject* view = nullptr;
    int rc = tpp_cuda_device_arg(arg, stream, view);
    if (rc < 0) {
      for (auto* v : views)
        Py_DECREF(v);
      Py_DECREF(launcher);
      return nullptr;
    }
    if (rc > 0) {
      largs[nconf + i] = view;
      views.push_back(view);
    }
  }

  PyObject* result =
      cpyrt_PyObject_Call(launcher, largs.data(), nconf + argc, kwds);
  for (auto* v : views)
    Py_DECREF(v);
  Py_DECREF(launcher);
  return result;
}

static PyObject* tpp_vectorcall(TemplateProxy* pytmpl, PyObject* const* args,
                                size_t nargsf, PyObject* kwds) {
  // Dispatcher to the actual member method, several uses possible; in order:
  //
  // case 1: explicit template previously selected through subscript
  //
  // case 2: select known non-template overload
  //
  //    obj.method(a0, a1, ...)
  //       => obj->method(a0, a1, ...)        // non-template
  //
  // case 3: select known template overload
  //
  //    obj.method(a0, a1, ...)
  //       => obj->method(a0, a1, ...)        // all known templates
  //
  // case 4: auto-instantiation from types of arguments
  //
  //    obj.method(a0, a1, ...)
  //       => obj->method<type(a0), type(a1), ...>(a0, a1, ...)
  //
  // Note: explicit instantiation needs to use [] syntax:
  //
  //    obj.method[type<a0>, type<a1>, ...](a0, a1, ...)
  //
  // case 5: low priority methods, such as ones that take void* arguments
  //

  // TODO: should previously instantiated templates be considered first?

  // case 0: CUDA kernel with a subscript-bound launch config
  if (pytmpl->fLaunchConfig)
    return tpp_cuda_launch(pytmpl, args, nargsf, kwds);

  PyObject *pymeth = nullptr, *result = nullptr;

  // short-cut through memoization map
  Py_ssize_t argc = cpyrt_PyArgs_GET_SIZE(args, nargsf);
  uint64_t sighash = HashSignature(args, argc);

  CPPOverload* ol = nullptr;
  if (!pytmpl->fTemplateArgs) {
    // look for known signatures ...
    auto& v = pytmpl->fTI->fDispatchMap[""];
    for (const auto& p : v) {
      if (p.first == sighash) {
        ol = p.second;
        break;
      }
    }

    if (ol != nullptr) {
      if (!pytmpl->fSelf || pytmpl->fSelf == Py_None) {
        result = cpyrt_tp_call((PyObject*)ol, args, nargsf, kwds);
      } else {
        pymeth = CPPOverload_Type.tp_descr_get((PyObject*)ol, pytmpl->fSelf,
                                               (PyObject*)&CPPOverload_Type);
        result = cpyrt_tp_call(pymeth, args, nargsf, kwds);
        Py_DECREF(pymeth);
        pymeth = nullptr;
      }
      if (result)
        return result;
    }
  }

  // container for collecting errors
  std::vector<Utility::PyError_t> errors;
  if (ol)
    Utility::FetchError(errors);

  // case 1: explicit template previously selected through subscript
  if (pytmpl->fTemplateArgs) {
    // instantiate explicitly
    PyObject* pyfullname =
        cpyrt_PyText_FromString(pytmpl->fTI->fCppName.c_str());
    cpyrt_PyText_Append(&pyfullname, pytmpl->fTemplateArgs);

    // first, lookup by full name, if previously stored
    bool isNS =
        (((CPPScope*)pytmpl->fTI->fPyClass)->fFlags & CPPScope::kIsNamespace);
    if (pytmpl->fSelf && pytmpl->fSelf != Py_None && !isNS)
      pymeth = PyObject_GetAttr(pytmpl->fSelf, pyfullname);
    else // by-passes custom scope getattr that searches into Cling
      pymeth = PyType_Type.tp_getattro(pytmpl->fTI->fPyClass, pyfullname);

    // attempt call if found (this may fail if there are specializations)
    if (CPPOverload_Check(pymeth)) {
      // since the template args are fully explicit, allow implicit conversion
      // of arguments
      result = CallMethodImp(pytmpl, pymeth, args, nargsf, kwds, /*impOK=*/true,
                             sighash);
      if (result) {
        Py_DECREF(pyfullname);
        TPPCALL_RETURN;
      }
      Utility::FetchError(errors);
    } else if (pymeth && PyCallable_Check(pymeth)) {
      // something different (user provided?)
      result = cpyrt_PyObject_Call(pymeth, args, nargsf, kwds);
      Py_DECREF(pymeth);
      if (result) {
        Py_DECREF(pyfullname);
        TPPCALL_RETURN;
      }
      Utility::FetchError(errors);
    } else if (!pymeth)
      PyErr_Clear();

    // not cached or failed call; try instantiation
    pymeth = pytmpl->Instantiate(cpyrt_PyText_AsString(pyfullname), args,
                                 nargsf, Utility::kNone);
    if (pymeth) {
      // attempt actual call; same as above, allow implicit conversion of
      // arguments
      result = CallMethodImp(pytmpl, pymeth, args, nargsf, kwds, /*impOK=*/true,
                             sighash);
      if (result) {
        Py_DECREF(pyfullname);
        TPPCALL_RETURN;
      }
    }

    // no drop through if failed (if implicit was desired, don't provide
    // template args)
    Utility::FetchError(errors);
    PyObject* topmsg = cpyrt_PyText_FromFormat(
        "Could not find \"%s\" (set cppjit.set_debug() for C++ errors):",
        cpyrt_PyText_AsString(pyfullname));
    Py_DECREF(pyfullname);
    Utility::SetDetailedException(std::move(errors), topmsg /* steals */,
                                  PyExc_TypeError /* default error */);

    return nullptr;
  }

  // case 2: select known non-template overload
  result = SelectAndForward(pytmpl, pytmpl->fTI->fNonTemplated, args, nargsf,
                            kwds, true /* implicitOkay */,
                            false /* use_targs */, sighash, errors);
  if (result)
    TPPCALL_RETURN;

  // case 3: select known template overload
  result = SelectAndForward(pytmpl, pytmpl->fTI->fTemplated, args, nargsf, kwds,
                            false /* implicitOkay */, true /* use_targs */,
                            sighash, errors);
  if (result)
    TPPCALL_RETURN;

  // case 4: auto-instantiation from types of arguments
  for (auto pref : {Utility::kReference, Utility::kPointer, Utility::kValue}) {
    // TODO: no need to loop if there are no non-instance arguments; also,
    // should any failed lookup be removed?
    int pcnt = 0;
    pymeth =
        pytmpl->Instantiate(pytmpl->fTI->fCppName, args, nargsf, pref, &pcnt);
    if (pymeth) {
      // attempt actual call; even if argument based, allow implicit
      // conversions, for example for non-template arguments
      result = CallMethodImp(pytmpl, pymeth, args, nargsf, kwds, /*impOK=*/true,
                             sighash);
      if (result)
        TPPCALL_RETURN;
    }
    Utility::FetchError(errors);
    if (!pcnt)
      break; // preference never used; no point trying others
  }

  // case 5: low priority methods, such as ones that take void* arguments
  result = SelectAndForward(pytmpl, pytmpl->fTI->fLowPriority, args, nargsf,
                            kwds, false /* implicitOkay */,
                            false /* use_targs */, sighash, errors);
  if (result)
    TPPCALL_RETURN;

  // error reporting is fraud, given the numerous steps taken, but more details
  // seems better
  if (!errors.empty()) {
    PyObject* topmsg =
        cpyrt_PyText_FromString("Template method resolution failed:");
    Utility::SetDetailedException(std::move(errors), topmsg /* steals */,
                                  PyExc_TypeError /* default error */);
  } else {
    PyErr_Format(PyExc_TypeError,
                 "cannot resolve method template call for \'%s\'",
                 pytmpl->fTI->fCppName.c_str());
  }

  return nullptr;
}

//----------------------------------------------------------------------------
static TemplateProxy* tpp_descr_get(TemplateProxy* pytmpl, PyObject* pyobj,
                                    PyObject*) {
  // create and use a new template proxy (language requirement)
  TemplateProxy* newPyTmpl =
      (TemplateProxy*)TemplateProxy_Type.tp_alloc(&TemplateProxy_Type, 0);

  // new method is to be bound to current object (may be nullptr)
  if (pyobj) {
    Py_INCREF(pyobj);
    newPyTmpl->fSelf = pyobj;
  } else {
    Py_INCREF(Py_None);
    newPyTmpl->fSelf = Py_None;
  }

  Py_XINCREF(pytmpl->fTemplateArgs);
  newPyTmpl->fTemplateArgs = pytmpl->fTemplateArgs;

  Py_XINCREF(pytmpl->fLaunchConfig);
  newPyTmpl->fLaunchConfig = pytmpl->fLaunchConfig;

  // copy name, class, etc. pointers
  new (&newPyTmpl->fTI) std::shared_ptr<TemplateInfo>{pytmpl->fTI};

  newPyTmpl->fVectorCall = pytmpl->fVectorCall;

  return newPyTmpl;
}

//----------------------------------------------------------------------------
static PyObject* tpp_subscript(TemplateProxy* pytmpl, PyObject* args) {
  // CUDA kernels take the launch config through the subscript; bind it
  // for the call to dispatch to the runtime launcher.
  if (tpp_is_cuda_kernel(pytmpl)) {
    PyObject* config = tpp_cuda_launch_config(args);
    if (!config)
      return nullptr;
    TemplateProxy* boundKernel = tpp_descr_get(pytmpl, pytmpl->fSelf, nullptr);
    Py_XDECREF(boundKernel->fLaunchConfig);
    boundKernel->fLaunchConfig = config;
    return (PyObject*)boundKernel;
  }

  // Explicit template member lookup/instantiation; works by re-bounding. This
  // method can not cache overloads as instantiations need not be unique for the
  // argument types due to template specializations.
  TemplateProxy* typeBoundMethod =
      tpp_descr_get(pytmpl, pytmpl->fSelf, nullptr);
  Py_XDECREF(typeBoundMethod->fTemplateArgs);
  typeBoundMethod->fTemplateArgs = cpyrt_PyText_FromString(
      Utility::ConstructTemplateArgs(nullptr, args).c_str());
  // Propagate the error that occurs if we can't construct the C++ name
  // from the provided template argument
  if (PyErr_Occurred()) {
    return nullptr;
  }
  return (PyObject*)typeBoundMethod;
}

//-----------------------------------------------------------------------------
static PyObject* tpp_getuseffi(CPPOverload*, void*) {
  return PyInt_FromLong(0); // dummy (__useffi__ unused)
}

//-----------------------------------------------------------------------------
static int tpp_setuseffi(CPPOverload*, PyObject*, void*) {
  return 0; // dummy (__useffi__ unused)
}

//----------------------------------------------------------------------------
static PyMappingMethods tpp_as_mapping = {nullptr, (binaryfunc)tpp_subscript,
                                          nullptr};

static PyGetSetDef tpp_getset[] = {
    {(char*)"__doc__", (getter)tpp_doc, (setter)tpp_doc_set, nullptr, nullptr},
    {(char*)"__useffi__", (getter)tpp_getuseffi, (setter)tpp_setuseffi,
     (char*)"unused", nullptr},
    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}};

//----------------------------------------------------------------------------
void TemplateProxy::Set(const std::string& cppname, const std::string& pyname,
                        PyObject* pyclass) {
  // Initialize the proxy for the given 'pyclass.'
  fSelf = nullptr;
  fTemplateArgs = nullptr;
  fLaunchConfig = nullptr;

  fTI->fCppName = cppname;
  Py_XINCREF(pyclass);
  fTI->fPyClass = pyclass;

  std::vector<PyCallable*> dummy;
  fTI->fNonTemplated = CPPOverload_New(pyname, dummy);
  fTI->fTemplated = CPPOverload_New(pyname, dummy);
  fTI->fLowPriority = CPPOverload_New(pyname, dummy);

  fVectorCall = (vectorcallfunc)tpp_vectorcall;
}

//= cpyrt method proxy access to internals ================================
static PyObject* tpp_overload(TemplateProxy* pytmpl, PyObject* args) {
  // Select and call a specific C++ overload, based on its signature.
  const char* sigarg = nullptr;
  PyObject* sigarg_tuple = nullptr;
  int want_const = -1;

  interop::TCppScope_t scope = nullptr;
  interop::TCppMethod_t cppmeth = nullptr;
  std::string proto;

  if (PyArg_ParseTuple(args, const_cast<char*>("s|i:__overload__"), &sigarg,
                       &want_const)) {
    want_const = PyTuple_GET_SIZE(args) == 1 ? -1 : want_const;

    // check existing overloads in order
    PyObject* ol = pytmpl->fTI->fNonTemplated->FindOverload(sigarg, want_const);
    if (ol)
      return ol;
    PyErr_Clear();
    ol = pytmpl->fTI->fTemplated->FindOverload(sigarg, want_const);
    if (ol)
      return ol;
    PyErr_Clear();
    ol = pytmpl->fTI->fLowPriority->FindOverload(sigarg, want_const);
    if (ol)
      return ol;

    proto = Utility::ConstructTemplateArgs(nullptr, args);
    // Propagate the error that occurs if we can't construct the C++ name
    // from the provided template argument
    if (PyErr_Occurred()) {
      return nullptr;
    }

    scope = ((CPPClass*)pytmpl->fTI->fPyClass)->fCppType;
    cppmeth = interop::GetMethodTemplate(scope, pytmpl->fTI->fCppName,
                                         proto.substr(1, proto.size() - 2));
  } else if (PyArg_ParseTuple(args, const_cast<char*>("O|i:__overload__"),
                              &sigarg_tuple, &want_const)) {
    PyErr_Clear();
    want_const = PyTuple_GET_SIZE(args) == 1 ? -1 : want_const;

    // check existing overloads in order
    PyObject* ol =
        pytmpl->fTI->fNonTemplated->FindOverload(sigarg_tuple, want_const);
    if (ol)
      return ol;
    PyErr_Clear();
    ol = pytmpl->fTI->fTemplated->FindOverload(sigarg_tuple, want_const);
    if (ol)
      return ol;
    PyErr_Clear();
    ol = pytmpl->fTI->fLowPriority->FindOverload(sigarg_tuple, want_const);
    if (ol)
      return ol;

    proto.reserve(128);
    proto.push_back('<');
    Py_ssize_t n = PyTuple_Size(sigarg_tuple);
    for (int i = 0; i < n; i++) {
      PyObject* pItem = PyTuple_GetItem(sigarg_tuple, i);
      if (!cpyrt_PyText_Check(pItem)) {
        PyErr_Format(PyExc_LookupError,
                     "argument types should be in string format");
        return (PyObject*)nullptr;
      }
      proto.append(cpyrt_PyText_AsString(pItem));
      if (i < n - 1)
        proto.push_back(',');
    }
    proto.push_back('>');

    scope = ((CPPClass*)pytmpl->fTI->fPyClass)->fCppType;
    cppmeth = interop::GetMethodTemplate(scope, pytmpl->fTI->fCppName,
                                         proto.substr(1, proto.size() - 2));
  } else {
    PyErr_Format(PyExc_TypeError, "Unexpected arguments to __overload__");
    return nullptr;
  }

  // else attempt instantiation
  if (!cppmeth) {
    return nullptr;
  }

  PyErr_Clear();

  // TODO: the next step should be consolidated with Instantiate()
  PyCallable* meth = nullptr;
  if (interop::IsNamespace(scope)) {
    meth = new CPPFunction(scope, cppmeth);
  } else if (interop::IsStaticMethod(cppmeth)) {
    meth = new CPPClassMethod(scope, cppmeth);
  } else if (interop::IsConstructor(cppmeth)) {
    meth = new CPPConstructor(scope, cppmeth);
  } else
    meth = new CPPMethod(scope, cppmeth);

  return (PyObject*)CPPOverload_New(pytmpl->fTI->fCppName + proto, meth);
}

static PyMethodDef tpp_methods[] = {{(char*)"__overload__",
                                     (PyCFunction)tpp_overload, METH_VARARGS,
                                     (char*)"select overload for dispatch"},
                                    {(char*)nullptr, nullptr, 0, nullptr}};

//= cpyrt template proxy type =============================================
PyTypeObject TemplateProxy_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type,
                          0)(char*) "cppjit.TemplateProxy", // tp_name
    sizeof(TemplateProxy),                                  // tp_basicsize
    0,                                                      // tp_itemsize
    (destructor)tpp_dealloc,                                // tp_dealloc
    offsetof(TemplateProxy, fVectorCall),
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_as_async / tp_compare
    0,                              // tp_repr
    0,                              // tp_as_number
    0,                              // tp_as_sequence
    &tpp_as_mapping,                // tp_as_mapping
    (hashfunc)tpp_hash,             // tp_hash
    (ternaryfunc)PyVectorcall_Call, // tp_call
    0,                              // tp_str
    0,                              // tp_getattro
    0,                              // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL |
        Py_TPFLAGS_METHOD_DESCRIPTOR,          // tp_flags
    (char*)"cppjit template proxy (internal)", // tp_doc
    (traverseproc)tpp_traverse,                // tp_traverse
    (inquiry)tpp_clear,                        // tp_clear
    (richcmpfunc)tpp_richcompare,              // tp_richcompare
    offsetof(TemplateProxy, fWeakrefList),     // tp_weaklistoffset
    0,                                         // tp_iter
    0,                                         // tp_iternext
    tpp_methods,                               // tp_methods
    0,                                         // tp_members
    tpp_getset,                                // tp_getset
    0,                                         // tp_base
    0,                                         // tp_dict
    (descrgetfunc)tpp_descr_get,               // tp_descr_get
    0,                                         // tp_descr_set
    0,                                         // tp_dictoffset
    0,                                         // tp_init
    0,                                         // tp_alloc
    (newfunc)tpp_new,                          // tp_new
    0,                                         // tp_free
    0,                                         // tp_is_gc
    0,                                         // tp_bases
    0,                                         // tp_mro
    0,                                         // tp_cache
    0,                                         // tp_subclasses
    0,                                         // tp_weaklist
    0,                                         // tp_del
    0,                                         // tp_version_tag
    0,                                         // tp_finalize
    0                                          // tp_vectorcall
    CPYRT_PYTYPE_TAIL};

} // namespace cppjit::cpyrt
