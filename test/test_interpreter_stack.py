import os

import cppjit
from pytest import mark, raises
from support import IS_CLANG_REPL

# A leaked interpreter would silently retarget every later test in this
# process, so the file refuses to start on a dirty stack.
assert cppjit.interpreter_stack_depth() == 0


def cplusplus():
    return cppjit.evaluate("__cplusplus")


class TestINTERPRETERSTACK:
    def test01_push_pop_restores_primary(self):
        """The primary interpreter is active again after a pop"""

        before = cplusplus()
        cppjit.push_interpreter("-std=c++23")
        assert cppjit.interpreter_stack_depth() == 1
        assert cplusplus() == 202302
        cppjit.pop_interpreter()
        assert cppjit.interpreter_stack_depth() == 0
        assert cplusplus() == before

    def test02_context_manager(self):
        """scoped_interpreter pops on normal exit"""

        with cppjit.scoped_interpreter("-std=c++23"):
            assert cppjit.interpreter_stack_depth() == 1
        assert cppjit.interpreter_stack_depth() == 0

    def test03_context_manager_pops_on_exception(self):
        """scoped_interpreter pops when the body raises"""

        class Marker(Exception):
            pass

        with raises(Marker):
            with cppjit.scoped_interpreter("-std=c++23"):
                raise Marker()
        assert cppjit.interpreter_stack_depth() == 0

    def test04_explicit_flags_beat_the_environment(self):
        """A scratch flag overrides CPPINTEROP_EXTRA_INTERPRETER_ARGS"""

        name = "CPPINTEROP_EXTRA_INTERPRETER_ARGS"
        saved = os.environ.get(name)
        os.environ[name] = "-std=c++17"
        try:
            with cppjit.scoped_interpreter("-std=c++23"):
                assert cplusplus() == 202302
        finally:
            if saved is None:
                del os.environ[name]
            else:
                os.environ[name] = saved

    def test05_nesting(self):
        """Interpreters nest strictly LIFO"""

        with cppjit.scoped_interpreter("-std=c++20"):
            assert cplusplus() == 202002
            with cppjit.scoped_interpreter("-std=c++23"):
                assert cppjit.interpreter_stack_depth() == 2
                assert cplusplus() == 202302
            assert cppjit.interpreter_stack_depth() == 1
            assert cplusplus() == 202002
        assert cppjit.interpreter_stack_depth() == 0

    def test06_pop_without_push(self):
        """Popping the primary interpreter is refused"""

        raises(RuntimeError, cppjit.pop_interpreter)
        assert cppjit.interpreter_stack_depth() == 0

    @mark.skipif(
        not IS_CLANG_REPL, reason="cling's interpreter ctor cannot report failure"
    )
    def test07_invalid_flag(self):
        """A rejected flag leaves the stack and the primary untouched"""

        raises(RuntimeError, cppjit.push_interpreter, "-std=c++42")
        assert cppjit.interpreter_stack_depth() == 0
        assert cppjit.cppdef(
            "namespace InterpStack { int after_failure() { return 7; } }"
        )
        assert cppjit.gbl.InterpStack.after_failure() == 7

    def test08_scratch_declarations(self):
        """Declarations made in a scope are usable through its own gbl"""

        with cppjit.scoped_interpreter("-std=c++23") as scope:
            assert cppjit.cppdef("""
            namespace InterpStackScratch {
            struct Value { int get(this Value& self) { return 21; } };
            }""")
            v = scope.InterpStackScratch.Value()
            assert v.get() == 21

    def test09_scratch_is_isolated_after_pop(self):
        """Scope declarations do not survive into the primary interpreter"""

        with cppjit.scoped_interpreter():
            assert cppjit.cppdef("namespace InterpStackGone { int fn() { return 1; } }")
        with raises(AttributeError):
            cppjit.gbl.InterpStackGone.fn()

    def test10_outer_entities_survive_a_scope(self):
        """The interpreter below a scope is untouched by it

        Its entities are called either side of the scope, never inside it,
        where their wrappers would compile against the pushed interpreter.
        """

        assert cppjit.cppdef("namespace InterpStackOuter { int fn() { return 5; } }")
        assert cppjit.gbl.InterpStackOuter.fn() == 5
        with cppjit.scoped_interpreter():
            assert cppjit.cppdef(
                "namespace InterpStackInner { int fn() { return 6; } }"
            )
        assert cppjit.gbl.InterpStackOuter.fn() == 5

    @mark.skipif(not IS_CLANG_REPL, reason="APINotes needs clang-repl")
    def test11_apinotes_flags(self):
        """API notes flags are accepted on a scoped interpreter"""

        with cppjit.scoped_interpreter("-fmodules", "-fapinotes-modules") as scope:
            assert cppjit.cppdef("namespace InterpStackNotes { int fn(){return 4;} }")
            assert scope.InterpStackNotes.fn() == 4
