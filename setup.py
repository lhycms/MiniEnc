from setuptools import find_packages, setup


setup(
    name="minienc",
    version="v1.0",
    author="Liu Hanyu",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        #"sklearn",
        "torch"
    ]
)