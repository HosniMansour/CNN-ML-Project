from setuptools import setup

setup(
    name='web',
    packages=['web'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)