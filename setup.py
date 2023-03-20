from setuptools import setup, find_packages
data_files_to_include = [('', ['README.md', 'LICENSE'])]

with open('requirements.txt', 'r') as reqs:
    requirements = reqs.read().split()

setup(name='tcr2vec',
      packages=find_packages(),
      version='1.0.0',
      description='TCR2vec is a deep representation learning framework of T-cell receptor sequence and function',
      long_description='Not applicable',
      url='https://github.com/jiangdada1221/TCR2vec',
      author='Yuepeng Jiang',
      author_email='jiangdada12344321@gmail.com',
      license='GPLv3',
      install_requires=requirements,
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            ],
      data_files = data_files_to_include,
      include_package_data=True,
      zip_safe=False)