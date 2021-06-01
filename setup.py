from setuptools import setup, find_packages

setup(
    name="fdrc",
    version="0.0.1",
    description="FDRC implementation and experiments",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)
