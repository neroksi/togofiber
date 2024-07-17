from setuptools import find_packages, setup


def do_setup():
    install_requires = [
        "scikit-learn==1.4.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "lightgbm==3.3.5",
        "catboost==1.2.5",
        "tqdm==4.65.0",
    ]

    extras = {}

    extras["test"] = []

    extras["all"] = sorted(
        set([rqrmt for _, flavour_rqrmts in extras.items() for rqrmt in flavour_rqrmts])
    )

    extras["dev"] = extras["all"]

    setup(
        name="togofiber",
        version="0.0.1",
        description="Togo Ministry Competition.",
        author="Kossi NEROMA",
        author_email="nkossy.pro@gmail.com",
        # url="#",
        packages=find_packages("src"),
        package_dir={"": "src"},
        # python_requires="==3.10.12",
        install_requires=install_requires,
        extras_require=extras,
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="OPTICAL FIBER TOGO",
    )


if __name__ == "__main__":
    do_setup()
