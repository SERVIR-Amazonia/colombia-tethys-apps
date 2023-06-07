from setuptools import setup, find_namespace_packages
from tethys_apps.app_installation import find_all_resource_files
from tethys_apps.base.app_base import TethysAppBase

# -- Apps Definition -- #
app_package = 'national_water_level_forecast_colombia'
release_package = f'{TethysAppBase.package_namespace}-{app_package}'

# -- Python Dependencies -- #
dependencies = []

# -- Get Resource File -- #
resource_files = find_all_resource_files(app_package, TethysAppBase.package_namespace)


setup(
    name=release_package,
    version='0.0.1',
    description='',
    long_description='',
    keywords='',
    author='Juseth Chancay, Jorge Luis Sanches',
    author_email='juseth.chancay@gmail.com, jlsanchezlo@unal.edu.co',
    url='',
    license='',
    packages=find_namespace_packages(),
    package_data={'': resource_files},
    include_package_data=True,
    zip_safe=False,
    install_requires=dependencies,
)