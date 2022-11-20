from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = 'intermittent-forecast',
    version = '0.0.1',
    description = 'Tools for forecasting intermittent time series',
    long_description = readme,
    long_description_content_type='text/markdown',
    author = 'Paul Morgan',
    author_email = '',
    url = '',
    python_requires = '>=3.8',
    install_requires = [
        'numpy>=1.20',
        'scipy>=1.6',
    ],
    license = license,
    packages = find_packages(where='src'),
    package_dir = {'':'src'},
)
