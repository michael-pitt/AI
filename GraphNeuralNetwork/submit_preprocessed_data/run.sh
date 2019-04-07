
INPUT_DATA="/mnt/lustre/agrp/pitt/ML/trackML/data/train_sample_single"

docker run -it --rm --memory="4g" --cpuset-cpus=0-1 \
	-v $(pwd):/home/code \
	-v $INPUT_DATA:/home/data \
	estradevictorantoine/trackml:1.0 \
	/bin/sh -c "cd /home/code; python setup.py build_ext --inplace; python main.py $*"
