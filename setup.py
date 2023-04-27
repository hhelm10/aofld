from setuptools import setup, find_packages

requirements = [
    "numpy",
    "joblib",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "seaborn"
]

with open("README.md", mode="r", encoding = "utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="aofld",
    version="0.0.1",
    author="MSR Special Projects & Johns Hopkins University",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages = ["aofld"],
    install_requires=requirements,
)
