from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fse-ops",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[
        CppExtension(
            name="fse_ops._C",
            sources=["src/fse_ops/ops.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
