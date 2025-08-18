from pytest import fixture, raises, mark
from pytest_lazy_fixtures import lf
from os.path import dirname, join

from pydmqmc.systems import System, Integral
from pydmqmc.methods import Method, AsymmetricBlochDMQMC
from pydmqmc import report_registry, enroll
from pydmqmc.report.report_functions import trace

default_enrolled =  ["trace", "energy", "von Neumann"]

class TestReportRegistry():

    @fixture
    def _setup(self, request):
        file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
        sys = Integral(file)
        sys.generate_hamiltonian()
        self._mtd = Method(sys)

    def test_len(self):
        assert len(report_registry) == len(default_enrolled)

    def test_contains(self):
        assert "trace" in report_registry

    def test_getitem(self):
        assert report_registry["trace"] is trace

    def test_keys(self):
        assert list(report_registry.keys()) == default_enrolled

    def test_list_requirements(self):
        assert report_registry.list_requirements("energy") == ("hamiltonian",)

    def test_list_requirements_none(self):
        assert report_registry.list_requirements("trace") is None

    #@mark.parametrize("func", default_enrolled)
    def test_get_requirements(self, _setup):
        req_dict = report_registry.get_requirements("energy", self._mtd)
        assert isinstance(req_dict, dict)
        assert "hamiltonian" in req_dict

    def test_get_requirements_bad(self):
        dummy_sys = System("/dummy/path")
        empty_mtd = Method(dummy_sys)
        with raises(RuntimeError):
            report_registry.get_requirements("energy", empty_mtd)

    def test_enroll_func(self, _setup):
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        report_registry.enroll("test",
                               test_func,
                               ["hamiltonian"])

        assert report_registry["test"] is test_func
        assert report_registry.get_requirements("test", self._mtd)

    def test_enroll_decorator_name_requires(self, _setup):
        @enroll(name="my_test", requires=["hamiltonian"])
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        assert report_registry["my_test"] is test_func
        assert report_registry.get_requirements("my_test", self._mtd)

    def test_enroll_decorator_requires_only(self, _setup):
        @enroll(requires=["hamiltonian"])
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        assert report_registry["test_func"] is test_func
        assert report_registry.get_requirements("test_func", self._mtd)

    def test_enroll_decorator_name_only(self, _setup):
        @enroll(requires=["hamiltonian"])
        def small_test(method):
            return method.system.input_file

        assert report_registry["small_test"] is small_test
        assert report_registry.get_requirements("small_test", self._mtd)
