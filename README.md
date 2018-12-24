## Baskets and Browsing sessions to vector (BB2vec)
## Complementary Products Recommendation

This is the source code of the paper "Inferring Complementary Products from Baskets and Browsing Sessions": https://arxiv.org/abs/1809.09621

Presentation: https://www.slideshare.net/ssuserb10599/inferring-complementary-products-from-baskets-and-browsing-sessions

## prerequisites
```
sudo apt-get install libboost-program-options1.55-dev;
```

## test framework installation
```
cd cxxtest/python;
sudo python setup.py install;
```

## build
```
make test;
make;
```
## usage

See https://github.com/IlyaTrofimov/bb2vec/blob/master/example.sh

Also you can check the possible options by 
./acc-rec --help

Baskets format: basket_id \t item_id

Views format: item1_id \t item2_id \t PMI
