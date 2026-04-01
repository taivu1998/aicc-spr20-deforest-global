from util import constants as C
from util import util as U
import argparse
from pathlib import Path

CHOSEN = ['ablation_050621_3e-5_1e-3_aggressive_False_none_True', 
          'ablation_050621_3e-5_1e-3_aggressive_False_onehot_False',
          'ablation_050621_3e-5_1e-3_aggressive_True_none_False',
          'ablation_050621_3e-5_1e-3_aggressive_False_none_False',
          'ablation_050621_3e-5_1e-3_aggressive_True_onehot_True'
         ]


def get_checkpoints(expt_path):
    ckpt_dir = expt_path / "ckpts"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), reverse=True)
    if ckpts:
        return ckpts

    legacy_ckpts = sorted(expt_path.glob("*.ckpt"), reverse=True)
    if legacy_ckpts:
        return legacy_ckpts

    raise FileNotFoundError(f"No checkpoints found under {ckpt_dir} or {expt_path}")

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for test-lpad func.')
    parser.add_argument('--prefix', '-p', help='prefix used when compiling lpad run', type=str)
    parser.add_argument('--split', '-s', help='Split', type=str, default='val')
    parser.add_argument('--num_expts', '-n', help='Max num expts to test', type=int, default=5)
    parser.add_argument('--chosen', '-c', help='Selectively run chosen expts', type=bool, default=False)

    args = parser.parse_args()

    if args.prefix is None: 
            raise Exception("Must enter prefix used for launchpad runs!")
    return args

if __name__ == "__main__":
    import main
    
    args = get_args()  
        
    df = U.get_tuning_metrics(prefix=args.prefix, save_path=None)
    print(f'===Generated results dataframe for expt prefix {args.prefix}===')
    print(df)
    
    for _, row in list(df.iterrows())[:args.num_expts]:
        expt_name = row['Model']
        if args.chosen and expt_name not in CHOSEN: 
            print(f'Skipping expt {expt_name}')
            continue
        expt_path = C.SANDBOX_DIR / expt_name
        ckpts = get_checkpoints(expt_path)
        print(f'Found {len(ckpts)} ckpts for expt {expt_name}')
        
        ckpt_path = ckpts[0] #single ckpt for now
        print(f'Testing ckpt {ckpt_path.stem} for expt {expt_path.stem}')
        main.test(ckpt_path=str(ckpt_path), test_split=args.split)
        
    print('Done!')
