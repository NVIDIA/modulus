from setuptools import setup, find_packages

setup(
    name='external_package',
    version='0.1',
    packages=find_packages(),
    entry_points={
        "modulus.models": [
            "CustomModel = external_package.custom_model:CustomModel",
        ]
    },
)
