brew upgrade cmake
brew upgrade gcc
brew install libomp

pipenv shell
git clone --recursive https://github.com/microsoft/LightGBM ; cd LightGBM
mkdir build ; cd build
cmake ..
make -j4

cd ../python-package/
python setup.py install --precompile
