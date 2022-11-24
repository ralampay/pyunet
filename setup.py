from setuptools import setup, find_packages

setup(
    name="PyUNet",
    version="0.1",
    description="Python UNet utility",
    author="Raphael Alampay",
    author_email="raphael.alampay@gmail.com",
    packages=find_packages("pyunet"),
    scripts=["bin/pyunet"]
)
