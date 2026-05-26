import os
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ('true', '1', 'yes', 'y'):
        return True
    if s in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected: {!r}'.format(v))


parser = argparse.ArgumentParser(description='DF-Net')

parser.add_argument('-ds', '--dataset', help='dataset, kvr or woz', required=False, default='kvr')
# parser.add_argument('-ds', '--dataset', help='dataset, kvr or woz', required=False, default='kvr')
parser.add_argument('-e', '--epoch', help='epoch num', required=False, type=int, default=1000)
parser.add_argument('-fixed', '--fixed', help='fix seeds (flag, no value)', action='store_true')
parser.add_argument('-random_seed', '--random_seed', help='random_seed', type=int, required=False, default=1234)
parser.add_argument('-em_dim', '--embeddings_dim', help='word embeddings dim', type=int, required=False, default=128)
parser.add_argument('-hdd', '--hidden', help='Hidden size', type=int, required=False, default=128)
parser.add_argument('-bsz', '--batch', help='Batch_size', type=int, required=False, default=32)
parser.add_argument('-lr', '--learn', help='Learning Rate', type=float, required=False, default=0.0001)
parser.add_argument('-dr', '--drop', help='Drop Out', type=float, required=False, default=0.3)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-gpu', '--gpu', help='use gpu (True/False)', type=_str2bool, required=False, default=True)
parser.add_argument('-l', '--layer', help='Layer Number', type=int, required=False, default=3)
parser.add_argument('-l_r', '--layer_r', help='RNN Layer Number', type=int, required=False, default=2)
parser.add_argument('-lm', '--limit', help='Word Limit', type=int, required=False, default=-10000)
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-clip', '--clip', help='gradient clipping', type=float, required=False, default=10)
parser.add_argument('-count', '--count', help='count for early stop', required=False, type=int, default=8)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                    default=0.9)

parser.add_argument('-evalp', '--evalp', help='evaluation period', type=int, required=False, default=1)
parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs', '--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='ENTF1')
parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                    default=1)
# parser.add_argument('-op', '--output', help='output file', required=False, default='kvr_log_cut/kvr_tha_ml27_cuttrain1.log')
parser.add_argument('-op', '--output', help='output file', required=False, default='kvr_log_whole/tha_ml29_768.log')
parser.add_argument(
    '-deepseek',
    '--use_deepseek_refinement',
    help='Use DeepSeek API to polish decoded responses during evaluate() only (needs DEEPSEEK_API_KEY)',
    action='store_true',
    default=False,
)
parser.add_argument(
    '-dnc',
    '--decode_n_candidates',
    type=int,
    default=1,
    help='evaluate(): sample this many stochastic decodes per example when >1; pick best by summed coarse-token log-prob before optional DeepSeek on the winner.',
)
parser.add_argument(
    '-dtemp',
    '--decode_temperature',
    type=float,
    default=0.8,
    help='Softmax temperature for multinomial coarse-token sampling when decode_n_candidates>1 (ignored when decode_n_candidates==1).',
)
parser.add_argument(
    '-dpw',
    '--decode_ptr_score_weight',
    type=float,
    default=1.0,
    help='When decode_n_candidates>1 and path score is used: add this weight * log P(memory slot | copy-type coarse token). Set 0 to use vocab-only score.',
)
parser.add_argument(
    '-loss_log',
    '--loss_log_csv',
    default='',
    type=str,
    help='If set, append one row per epoch with mean train losses (loss, loss_g, loss_v, loss_l)',
)
parser.add_argument(
    '-than_out',
    '--than_attn_outdir',
    default='',
    type=str,
    help='If set, save THAN heatmaps (epochXXX_all.png, epochXXX_group_integrated_fusion.png, ...) each period',
)
parser.add_argument(
    '-than_ev',
    '--than_attn_every',
    default=0,
    type=int,
    help='Save THAN snapshots every N epochs; 0 means use -evalp as period',
)
parser.add_argument(
    '-than_mb',
    '--than_attn_max_batches',
    default=40,
    type=int,
    help='Max train batches to average for one THAN snapshot',
)
parser.add_argument(
    '-trace_csv',
    '--dev_trace_csv',
    default='',
    type=str,
    help='If set, append epoch,alpha,beta,dev_score after each dev evaluate (for plot_alpha_beta_trace.py)',
)
parser.add_argument(
    '-vis_metrics_csv',
    '--vis_metrics_csv',
    default='',
    type=str,
    help='If set, append epoch,dev_score after each dev evaluate (for plot_dialogue_metrics_csv.py)',
)

parser.add_argument(
    '-than_xtick_max',
    '--than_xtick_max',
    type=int,
    default=0,
    help='THAN heatmaps: max number of x-axis token labels to print (rest blank); 0 = show all tokens (default)',
)
parser.add_argument(
    '-than_highlight',
    '--than_highlight',
    default='',
    type=str,
    help='THAN heatmaps: comma-separated tokens/substrings to draw blue column boxes (e.g. need,stay,alexander_bed_and_breakfast)',
)
parser.add_argument(
    '--train_file',
    default='',
    type=str,
    help='Override train.txt path (default: dataset standard). Unset = original full train.',
)
parser.add_argument(
    '--dev_file',
    default='',
    type=str,
    help='Override dev.txt path (default: dataset standard).',
)
parser.add_argument(
    '--test_file',
    default='',
    type=str,
    help='Override test.txt path (default: dataset standard). Test is not subsampled by target_ratio.',
)
parser.add_argument(
    '-td',
    '--target_domain',
    default='',
    type=str,
    help='Few-shot target domain (kvr: navigate|weather|schedule; woz: restaurant|hotel|attraction). '
         'Source domains stay 100%%; target domain keeps --target_ratio of **whole dialogues** (all turns).',
)
parser.add_argument(
    '--target_ratio',
    type=float,
    default=1.0,
    help='With -td: fraction of target-domain train dialogues only (other domains 100%%). '
         'Dev/test unchanged. Do not combine with --train_ratio unless you intend both.',
)
parser.add_argument(
    '--train_ratio',
    type=float,
    default=1.0,
    help='All domains: fraction of train **dialogues** to keep (e.g. 0.1, 0.3, 0.5). No -td needed. '
         'Dev/test stay full. Whole-dialogue sampling; all turns in a kept dialogue retained.',
)
parser.add_argument(
    '-xd',
    '--exclude_domains',
    default='',
    type=str,
    help='Comma-separated domains removed entirely from train/dev (zero-shot on target).',
)
parser.add_argument(
    '--dev_exclude_target',
    action='store_true',
    help='Remove all -target_domain samples from dev (early stop without target-domain leakage).',
)

args = vars(parser.parse_args())
print(str(args))
USE_CUDA = args['gpu']
print("USE_CUDA: " + str(USE_CUDA))

LIMIT = int(args["limit"])
MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 12


class DeepSeekConfig:
    """DeepSeek OpenAI-compatible API (used only when -deepseek is set)."""
    BASE_URL = "https://api.deepseek.com"
    MODEL_NAME = "deepseek-chat"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
