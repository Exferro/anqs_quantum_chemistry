from setuptools import setup

setup(
   name='nqs',
   version='0.1',
   description='A custom NQS library',
   author='Aleksei Malyshev',
   author_email='aleksei.o.malyshev@gmail.com',
   packages=['nqs'],  #same as name
   install_requires=['wheel'], #external packages as dependencies
)
