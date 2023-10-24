import os
import os.path as osp
import argparse

from utils import prompt_llm, prompt_llm_rank

# Default directories
ABS_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(ABS_DIR, '../../data')
CONFIG_DIR = osp.join(ABS_DIR, '../../config')
PROMPT_DIR = osp.join(ABS_DIR, '../../prompts')
PROMPT_OUTDIR = osp.join(ABS_DIR, '../../prompt_outputs')

def main(args):
    for datapred in args.datapreds:
        dataset, pred = datapred.split('/')

        if args.rank:
            _ = prompt_llm_rank(
                dataset=dataset,
                pred=pred,
                llm_model=args.llm_model,
                temperature=args.temperature,
                n_samples=args.n_samples,
                max_tokens=1024,
                verbose=args.verbose
            )

        else:
            _ = prompt_llm(
                dataset=dataset,
                pred=pred,
                llm_model=args.llm_model,
                min_score=args.min_score,
                max_score=args.max_score,
                temperature=args.temperature,
                n_samples=args.n_samples,
                max_tokens=256,
                batch_size=args.batch_size,
                verbose=args.verbose,
                add_context=args.add_context,
                add_examples=args.add_examples,
                add_expls=args.add_expls,
                query=args.query
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapreds', help='Dataset + prediction task pair', default=['mimic/sepsis'], nargs='*', type=str)
    parser.add_argument('--llm_model', help='LLM model to prompt', default='llama-2-13b-chat', type=str)
    parser.add_argument('--min_score', help='Minimum feature importance score', default=0, type=float)
    parser.add_argument('--max_score', help='Maximum feature importance score', default=1, type=float)
    parser.add_argument('--temperature', help='Temperature setting for prompting', default=0, type=float)
    parser.add_argument('--n_samples', help='Number of LLM samples', default=1, type=int)
    parser.add_argument('--batch_size', help='Batch size for multiple prompts (Llama-2 only)', default=128, type=int)
    parser.add_argument('--add_context', help='Option to add context', default=False, action='store_true')
    parser.add_argument('--add_examples', help='Option to add examples', default=False, action='store_true')
    parser.add_argument('--add_expls', help='Option to add reasoning in few-shot examples', default=False, action='store_true')
    parser.add_argument('--query', help='Concepts to selectively query', default=None, nargs='*', type=str)
    parser.add_argument('--rank', help='Option to generate direct rankings', default=False, action='store_true')
    parser.add_argument('--verbose', help='Output verbosity', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
