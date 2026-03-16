from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="zipf_encoding",
    version="0.1.0",
    rust_extensions=[
        RustExtension("zipf_encoding.zipf_encoding", binding=Binding.PyO3)
    ],
    packages=["zipf_encoding"],
    # zip_safe flag can be set to False if you want to load the extension from inside the ZIP file
    zip_safe=False,
)
