dirs:
	mkdir -p data/raw
	mkdir -p data/preprocess
	mkdir -p data/models
download: dirs
	wget -O data/raw/b3db.tsv https://github.com/theochem/B3DB/blob/main/raw_data/raw_data_summary.tsv

features:
	python features.py