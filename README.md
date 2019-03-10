# Automatic aligmement and color composition of picture

This project is a Student project, the goal is to merge Sergey Prokudin-Gorsky pictures in one colored picture.
Sergey Prokudin-Gorsky  use to take grayscale photo with colored filted to get the 3 colors channels. 

# Get started

You will need python3, opencv, numpy, pillow

Donwload images from [Library of Congress](https://www.loc.gov/pictures/search/?q=Prokudin+negative&sp=2&st=grid) like : 

- [1](http://cdn.loc.gov/master/pnp/prok/00400/00451u.tif)
- [2](http://cdn.loc.gov/master/pnp/prok/00900/00998u.tif)
- [3](http://cdn.loc.gov/master/pnp/prok/01500/01520u.tif)
 

```bash

git clone https://github.com/nealith/im.git
cd im
python program.py -i 00451u.tif -o 00451u_composed.png
```
see option -h for help
