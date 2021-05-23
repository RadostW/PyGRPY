from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='PyGRPY',
      version='0.0.2',
      description='Python port of Generalized Rotne Prager Yakamava tensors',
      url='https://github.com/RadostW/PyGYPY/',
      author='Radost Waszkiewicz',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      license='GNU GPLv3',
      packages=['pygrpy'],
      zip_safe=False)
