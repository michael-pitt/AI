# Instructions - how to generate data for training:

if running on a public cluster you can do the following:
```bash
INPUT_DATA=INPUTDATA
docker run -it -p 8888:8888 --rm -v $(pwd):/home/code -v $INPUT_DATA:/home/data estradevictorantoine/trackml:1.0
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

in a new shell open a tunnel
```bash
ssh -N -f -L localhost:7008:localhost:8888 user@adress.of.the.cluster
```

then in a browser open: `http://localhost:7008/XXXXXXXXXX`
