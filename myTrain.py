from tqdm import tqdm
import csv
import os
import sys

from utils.config import *
from models.model import *

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CUDA_LAUNCH_BLOCKING=1

# fixed random seed
if args['fixed']:
    torch.manual_seed(args['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        torch.backends.cudnn.deterministic = True
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])

# load data process function
early_stop = args['earlyStop']
if args['dataset'] == 'kvr':
    from utils.utils_Ent_kvr import *
    # domains = {['navigate','weather']: 0, ['schedule','weather']: 1, ['navigate','schedule']: 2}
    domains = {'navigate': 0, 'weather': 1, 'schedule': 2}
elif args['dataset'] == 'woz':
    from utils.utils_Ent_woz import *
    domains = {'restaurant': 0, 'attraction': 1, 'hotel': 2}
else:
    print("[ERROR] You need to provide the correct --dataset information")

# Configure models and load data
if args['epoch'] > 0:
    avg_best, cnt, res = 0.0, 0, 0.0
    train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))
    model = globals()['DFNet'](
        int(args['hidden']),
        lang,
        max_resp_len,
        args['path'],
        lr=float(args['learn']),
        n_layers=int(args['layer']),
        dropout=float(args['drop']),
        domains=domains)

    # Training
    for epoch in range(args['epoch']):
        print("Epoch:{}".format(epoch))
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data, int(args['clip']), reset=(i == 0))
            pbar.set_description(model.print_loss())
        than_out = (args.get('than_attn_outdir') or '').strip()
        if than_out:
            than_every = int(args.get('than_attn_every') or 0)
            period = than_every if than_every > 0 else int(args['evalp'])
            if (epoch + 1) % period == 0:
                from utils.than_attention_viz import save_than_training_snapshot

                mb = int(args.get('than_attn_max_batches', 40))
                path = save_than_training_snapshot(model, train, lang, than_out, epoch + 1, max_batches=mb)
                if path:
                    print(f"[THAN] Saved attention heatmaps -> {path}")
                else:
                    print("[THAN] Attention snapshot skipped (no weights collected)")
        if (epoch + 1) % int(args['evalp']) == 0:
            res = model.evaluate(dev, avg_best, early_stop=early_stop)
            model.scheduler.step(res)
            trace_path = (args.get('dev_trace_csv') or '').strip()
            if trace_path:
                row_t = {
                    'epoch': epoch + 1,
                    'alpha': float(args.get('alpha', 1.0)),
                    'beta': float(args.get('beta', 1.0)),
                    'dev_score': float(res),
                }
                new_t = not os.path.exists(trace_path)
                dt = os.path.dirname(os.path.abspath(trace_path))
                if dt:
                    os.makedirs(dt, exist_ok=True)
                with open(trace_path, 'a', newline='', encoding='utf-8') as tf:
                    wt = csv.DictWriter(tf, fieldnames=['epoch', 'alpha', 'beta', 'dev_score'])
                    if new_t:
                        wt.writeheader()
                    wt.writerow(row_t)
            vis_m = (args.get('vis_metrics_csv') or '').strip()
            if vis_m:
                row_m = {'epoch': epoch + 1, 'dev_score': float(res)}
                new_m = not os.path.exists(vis_m)
                dm = os.path.dirname(os.path.abspath(vis_m))
                if dm:
                    os.makedirs(dm, exist_ok=True)
                with open(vis_m, 'a', newline='', encoding='utf-8') as mf:
                    wm = csv.DictWriter(mf, fieldnames=['epoch', 'dev_score'])
                    if new_m:
                        wm.writeheader()
                    wm.writerow(row_m)
            if res >= avg_best:
                avg_best = res
                cnt = 0
            else:
                cnt += 1
            if cnt == args['count']:
                print("Ran out of patient, early stop...")
                break
        if args.get('loss_log_csv'):
            path = args['loss_log_csv']
            nb = max(len(train), 1)
            row = {
                'epoch': epoch,
                'loss': model.loss / nb,
                'loss_g': model.loss_g / nb,
                'loss_v': model.loss_v / nb,
                'loss_l': model.loss_l / nb,
            }
            new_file = not os.path.exists(path)
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, 'a', newline='', encoding='utf-8') as lf:
                w = csv.DictWriter(lf, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)

# Testing
train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))

model = globals()['DFNet'](
    int(args['hidden']),
    lang,
    max_resp_len,
    args['path'],
    lr=0.0,
    n_layers=int(args['layer']),
    dropout=0.0,
    domains=domains)

_than_out = (args.get('than_attn_outdir') or '').strip()
_p = args.get('path') or ''
if _than_out and int(args.get('epoch', 0)) == 0:
    if not _p or not os.path.exists(str(_p)):
        print("[THAN] Skip heatmap: with -e 0 you need a valid -path to trained save/... (enc.th, dec.th, enc_kb.th)")
    else:
        from utils.than_attention_viz import save_than_training_snapshot

        _mb = int(args.get('than_attn_max_batches', 40))
        _path = save_than_training_snapshot(
            model,
            train,
            lang,
            _than_out,
            epoch=0,
            max_batches=_mb,
            snapshot_prefix='from_checkpoint',
        )
        if _path:
            print(f"[THAN] Saved attention heatmaps (no training, loaded weights) -> {_path}")
        else:
            print("[THAN] Attention snapshot skipped (no weights collected)")

#max_resp_len可能需要修改

res_test = model.evaluate(test, 1e7, output=True)

# THA:-gpu=True
# -ds=woz
# -dr=0.2
# -bsz=32
# -tfr=0.9
# -an=WOZ_THA
# -op=WOZ.log

