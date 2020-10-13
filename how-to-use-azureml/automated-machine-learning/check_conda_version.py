from distutils.version import LooseVersion
import platform

try:
    import conda
except:
    print('Failed to import conda.')
    print('This setup is usually run from the base conda environment.')
    print('You can activate the base environment using the command "conda activate base"')
    exit(1)

architecture = platform.architecture()[0]

if architecture != "64bit":
    print('This setup requires 64bit Anaconda or Miniconda.  Found: ' + architecture)
    exit(1)

minimumVersion = "4.7.8"

versionInvalid = (LooseVersion(conda.__version__) < LooseVersion(minimumVersion))

if versionInvalid:
    print('Setup requires conda version ' + minimumVersion + ' or higher.')
    print('You can use the command "conda update conda" to upgrade conda.')

exit(versionInvalid)
