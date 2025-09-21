import sys
from data_util import *

sys.path.append('../util/*')
from constants import *

def main():
	for split in ['train', 'val', 'test']:
		old_path = PREDOWNLOAD_DIR / f'{split}_latlon_v3_with_lossyear.csv'
		new_path = DATA_BASE_DIR / f'predownload_meta_{split}.csv'
		indo_to_hansen_download_meta(old_path, new_path, split)
		print(f"Finished fixing split {split}")

if __name__ == "__main__":
	main()