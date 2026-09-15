from setuptools import find_namespace_packages, setup


setup(
    name="dynaslots",
    version="0.1.0",
    description="Dynamic object-centric 3D representations for robot learning",
    packages=find_namespace_packages(include=("dynaslots", "dynaslots.*")),
    package_data={
        "dynaslots": ["config/*.yaml", "config/task/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.8",
)
