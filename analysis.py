#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict

def compute_success_rate(results_file: str) -> float:
    with open(os.path.join('results', results_file), 'r') as f:
        data = json.load(f)
    if not data:
        return 0.0
    last_iteration = data[-1]
    total = len(last_iteration)
    if total == 0:
        return 0.0
    successes = sum(1 for v in last_iteration.values() if v['success'])
    return successes / total

def generate_table(rows_info: list[dict], output_file: str):
    # Compute success rates and group by model
    rows = []
    for row in rows_info:
        results_file = row['results_file']
        model_name = row['model_name']
        finetune_data = row['finetune_data']
        success_rate = compute_success_rate(results_file)
        rows.append({
            'model_name': model_name,
            'finetune_data': finetune_data,
            'success_rate': success_rate
        })

    # Group rows by model to determine where to place \midrule
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row['model_name']].append(row)

    # Generate LaTeX table
    with open(output_file, 'w') as f:
        f.write('\\begin{tabular}{lcc}\n')
        f.write('\\hline\n')
        f.write('\\textbf{Model} & \\textbf{Fine-tuning data} & \\textbf{Success Rate} \\\\\n')
        f.write('\\hline\n')

        for idx, (model_name, model_rows) in enumerate(grouped_rows.items()):
            for i, row in enumerate(model_rows):
                model_cell = model_name if i == 0 else ''
                finetune_data = row['finetune_data']
                success_rate = f"{row['success_rate']*100:.1f}\\%"
                f.write(f"{model_cell} & {finetune_data} & {success_rate} \\\\\n")
            if idx < len(grouped_rows) - 1:
                f.write('\\midrule\n')

        f.write('\\hline\n')
        f.write('\\end{tabular}\n')

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX table from results.')
    parser.add_argument('--table', action='store_true', help='Generate LaTeX table.')
    parser.add_argument('--table-rows', type=str, required=True, help='Path to JSON file describing the table rows.')
    parser.add_argument('--output', type=str, required=True, help='Name of the LaTeX file to write.')
    args = parser.parse_args()

    if args.table:
        # Read table rows info
        with open(args.table_rows, 'r') as f:
            rows_info = json.load(f)
        generate_table(rows_info, args.output)
    else:
        print('No action specified. Use --table to generate the table.')

if __name__ == '__main__':
    main()
