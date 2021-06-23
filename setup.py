from setuptools import setup


def get_version(filename: str):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename="duckietown_rl/__init__.py")


setup(
    name="duckietown_rl",
    version=version,
    packages=["duckietown_rl"],
    install_requires=[
        "gym==0.15.4",
        "gym_duckietown_agent>=2018.08",
        "hyperdash",  # for logging
        "sklearn",
        "torch",
        "numpy",
        "matplotlib",
        "scipy<=1.2.1",
    ],
)
