from setuptools import  find_packages, setup
HYPHEN_E_DOT = '-e .'

# This is the setup file for the ml-project package. It will be used to install the package and its dependencies.
def get_requirements(file_path)-> list[str]:
    """ This function will return the list of requirements"""
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
    requirements = [req.replace('\n', '') for req in requirements]
    if HYPHEN_E_DOT in requirements:    
     requirements.remove(HYPHEN_E_DOT)
     return requirements

setup (
name='ml-project',  
version='0.0.1',
author="Mahwish",
author_email="mehwishh165@gmail.com",
packages=find_packages(),
install_requires=get_requirements('requirements.txt')  )
    
                 