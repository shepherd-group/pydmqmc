from pytest import fixture
from os.path import dirname, join

from pydmqmc.systems import Integral
from pydmqmc.methods import AsymmetricBlochDMQMC
from pydmqmc import report_registry, enroll
from pydmqmc.report.report_functions import trace

@fixture
def default_enrolled():
    return ["trace"]

class TestReportRegistry():

    @fixture
    def _setup(self, request):
        file = join(dirname(request.path),
                "..", "inputs", "integrals", "H2-STO-3G-0.74Ang.fcidump")
        sys = Integral(file)
        self._mtd = AsymmetricBlochDMQMC(sys)
        self._mtd.setup("deterministic")

    def test_len(self, default_enrolled):
        assert len(report_registry) == len(default_enrolled)

    def test_contains(self):
        assert "trace" in report_registry

    def test_getitem(self):
        assert report_registry["trace"] is trace

    def test_keys(self, default_enrolled):
        assert list(report_registry.keys()) == default_enrolled

    def test_check_requirements(self, _setup):
        assert report_registry.check_requirements("trace", self._mtd)

    def test_enroll_func(self, _setup):
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        report_registry.enroll("test",
                               test_func,
                               ["hamiltonian", "density_matrix"])

        assert report_registry["test"] is test_func
        assert report_registry.check_requirements("test", self._mtd)

    def test_enroll_decorator_name_requires(self, _setup):
        @enroll(name="my_test", requires=["hamiltonian", "density_matrix"])
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        assert report_registry["my_test"] is test_func
        assert report_registry.check_requirements("my_test", self._mtd)

    def test_enroll_decorator_requires_only(self, _setup):
        @enroll(requires=["hamiltonian", "density_matrix"])
        def test_func(method):
            p = method.density_matrix
            H = method.system.hamiltonian
            return (p @ H).trace()

        assert report_registry["test_func"] is test_func
        assert report_registry.check_requirements("test_func", self._mtd)

    def test_enroll_decorator_name_only(self, _setup):
        @enroll(requires=["hamiltonian", "density_matrix"])
        def small_test(method):
            return method.system.input_file

        assert report_registry["small_test"] is small_test
        assert report_registry.check_requirements("small_test", self._mtd)
