from setuptools import setup, find_packages
import os, sys
from pathlib import Path
sys.path.insert(0, f'{os.path.dirname(__file__)}{os.sep}mousechd')
import mousechd
version = mousechd.__version__
this_dir = Path(__file__).parent
long_description = (this_dir/"README.md").read_text()

setup(name='mousechd',
      version=version,
      python_requires=">=3.9",
      description='Segmenting hearts and screening congenital heart diseases in mice',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/hnguyentt/MouseCHD",
      author='Hoa Nguyen',
      author_email='ntthoa.uphcm@gmail.com',
      license="MIT",
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              "mousechd = mousechd.__main__:main"
          ]
      },
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "Intended Audience :: Healthcare Industry",
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Scientific/Engineering :: Image Recognition",
                   "Topic :: Scientific/Engineering :: Medical Science Apps.",
                   ],
      keywords=["deep learning",
                "image segmentation",
                "image classification",
                "medical image analysis",
                "mousechd"],
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
          "tensorflow==2.14.0",
          "brokenaxes"
        #   "napari",
        #   "PyQt5"
      ]
      )