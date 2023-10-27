from setuptools import setup, find_packages
import os, sys
sys.path.insert(0, f'{os.path.dirname(__file__)}{os.sep}mousechd')
import mousechd
version = mousechd.__version__

setup(name='mousechd',
      version=version,
      description='Segmenting hearts and screening congenital heart diseases in mice',
      author='IMOD Pasteur',
      author_email='hnguyent@pasteur.fr | hoantt.d2011@uphcm.edu.vn',
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              "mousechd = mousechd.__main__:main"
          ]
      },
      install_requires=[
          "opencv-contrib-python-headless",
          "SimpleITK",
          "tifffile",
          "matplotlib",
          "seaborn",
          "venn",
          "pydicom",
          "pynrrd",
          "PyYAML",
          "nnunet==1.7.1",
          "volumentations-3D",
          "tensorflow",
          "napari",
          "PyQt5"
      ]
      )