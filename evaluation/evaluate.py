"""Strict decision evaluation. Standard library only; no inference or dataset edits."""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

DECISIONS = ('continue', 'pause', 'cancel', 'replace_task', 'clarify')
SKILLS = ('follow_player', 'return_home', 'collect', 'craft', 'store_items')


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite JSON number: ' + value)


def parse_output(raw):
    """Reject prose, fences, duplicate/extra keys, and invalid skill semantics."""
    if not isinstance(raw, str):
        return None, 'output_not_string'
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return None, 'invalid_json'
    if not isinstance(value, dict) or set(value) != {'decision', 'skill'}:
        return None, 'invalid_keys_or_object'
    if not isinstance(value['decision'], str) or value['decision'] not in DECISIONS:
        return None, 'invalid_decision'
    if value['decision'] == 'replace_task':
        if not isinstance(value['skill'], str) or value['skill'] not in SKILLS:
            return None, 'replacement_requires_known_skill'
    elif value['skill'] is not None:
        return None, 'nonreplacement_requires_null_skill'
    return value, None


def load_gold(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Gold must be a nonempty JSON array.')
    seen = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or row['id'] in seen:
            raise ValueError('Missing or duplicate gold id.')
        seen.add(row['id'])
        if not isinstance(row.get('family'), str) or not row['family']:
            raise ValueError('Missing gold family: ' + row['id'])
        _, error = parse_output(row['output'])
        if error:
            raise ValueError('Invalid gold ' + row['id'] + ': ' + error)
    return rows


def load_predictions(path):
    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            except ValueError as error:
                raise ValueError(f'Invalid prediction JSONL envelope at line {line_number}: {error}') from error
            if not isinstance(row, dict):
                raise ValueError(f'Prediction envelope at line {line_number} is not an object.')
            rows.append(row)
    return rows


def align(gold, predictions):
    by_id = {}
    for pred in predictions:
        identifier = pred.get('id')
        if not isinstance(identifier, str) or identifier in by_id:
            raise ValueError('Missing or duplicate prediction id: ' + str(identifier))
        by_id[identifier] = pred
    gold_ids = {r['id'] for r in gold}
    extra = set(by_id) - gold_ids
    if extra:
        raise ValueError('Unknown prediction ids: ' + ', '.join(sorted(extra)[:5]))
    aligned = []
    for row in gold:
        pred = by_id.get(row['id'])
        raw_available = pred is not None and isinstance(pred.get('raw_output'), str)
        if pred is None:
            parsed, error = None, 'missing_prediction'
        elif 'raw_output' in pred:
            parsed, error = parse_output(pred['raw_output'])
        elif 'prediction' in pred:
            supplied = pred['prediction']
            parsed, error = parse_output(supplied if isinstance(supplied, str) else json.dumps(supplied, ensure_ascii=False))
        else:
            parsed, error = None, 'missing_output'
        latency = None if pred is None else pred.get('latency_seconds', pred.get('latency'))
        if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0):
            raise ValueError('Invalid latency_seconds: ' + row['id'])
        aligned.append({'id': row['id'], 'family': row['family'], 'gold': json.loads(row['output']),
                        'prediction': parsed, 'error': error, 'raw_available': raw_available, 'latency_seconds': latency})
    return aligned


def ratio(numerator, denominator):
    return {'value': numerator / denominator if denominator else None, 'numerator': numerator, 'denominator': denominator}


def summarize(aligned):
    n = len(aligned)
    confusion = {label: {p: 0 for p in (*DECISIONS, 'invalid')} for label in DECISIONS}
    errors = Counter()
    decision_correct = joint_correct = valid = raw_valid = raw_available = 0
    skill_correct = skill_denominator = cancel_fp = cancel_fn = cancel_gold = noncancel_gold = 0
    latencies = []
    for row in aligned:
        gold, pred = row['gold'], row['prediction']
        p = pred['decision'] if pred else 'invalid'
        g = gold['decision']
        confusion[g][p] += 1
        valid += pred is not None
        raw_available += row['raw_available']
        raw_valid += bool(row['raw_available'] and pred is not None)
        decision_correct += p == g
        joint_correct += pred == gold
        if row['error']:
            errors[row['error']] += 1
        if g == 'replace_task':
            skill_denominator += 1
            skill_correct += bool(pred and pred['skill'] == gold['skill'])
        if g == 'cancel':
            cancel_gold += 1
            cancel_fn += p != 'cancel'
        else:
            noncancel_gold += 1
            cancel_fp += p == 'cancel'
        if row['latency_seconds'] is not None:
            latencies.append(row['latency_seconds'])
    per_class = {}
    for label in DECISIONS:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in DECISIONS if g != label)
        fn = sum(count for p, count in confusion[label].items() if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    return {
        'n': n, 'families': len({r['family'] for r in aligned}),
        'decision_accuracy': decision_correct / n if n else None,
        'joint_accuracy': joint_correct / n if n else None,
        'macro_f1': sum(p['f1'] for p in per_class.values()) / len(DECISIONS),
        'strict_json_validity': raw_valid / n if n else None,
        'json_validity': raw_valid / n if n else None,
        'schema_validity': valid / n if n else None,
        'raw_output_coverage': ratio(raw_available, n),
        'strict_json_counts': ratio(raw_valid, n),
        'skill_accuracy': skill_correct / skill_denominator if skill_denominator else None,
        'skill_accuracy_counts': ratio(skill_correct, skill_denominator),
        'false_cancel_rate': ratio(cancel_fp, noncancel_gold),
        'missed_cancel_rate': ratio(cancel_fn, cancel_gold),
        'decision_accuracy_counts': ratio(decision_correct, n),
        'joint_accuracy_counts': ratio(joint_correct, n),
        'invalid_or_missing': dict(errors), 'per_class': per_class, 'confusion': confusion,
        'latency_seconds': {'n': len(latencies), 'mean': sum(latencies) / len(latencies) if latencies else None,
                            'median': _percentile(sorted(latencies), .5) if latencies else None,
                            'p95': _percentile(sorted(latencies), .95) if latencies else None},
    }


def _percentile(sorted_values, probability):
    if not sorted_values:
        return None
    index = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return sorted_values[lower] * (upper - index) + sorted_values[upper] * (index - lower) if lower != upper else sorted_values[lower]


def paired_family_bootstrap(left, right, iterations=2000, seed=20261029):
    """Resample whole families; report right-minus-left metric differences."""
    if iterations < 1:
        raise ValueError('Bootstrap iterations must be positive.')
    if [(r['id'], r['family'], r['gold']) for r in left] != [(r['id'], r['family'], r['gold']) for r in right]:
        raise ValueError('Paired bootstrap requires identical aligned gold order.')
    groups = defaultdict(list)
    for i, row in enumerate(left):
        groups[row['family']].append(i)
    families = sorted(groups)
    if not families:
        raise ValueError('No families to bootstrap.')
    rng = random.Random(seed)
    metrics = ('decision_accuracy', 'joint_accuracy', 'macro_f1')
    differences = {key: [] for key in metrics}
    for _ in range(iterations):
        indices = [i for family in rng.choices(families, k=len(families)) for i in groups[family]]
        a = summarize([left[i] for i in indices]); b = summarize([right[i] for i in indices])
        for key in metrics:
            differences[key].append(b[key] - a[key])
    observed_a, observed_b = summarize(left), summarize(right)
    return {'direction': 'comparison_minus_primary', 'unit': 'family', 'seed': seed, 'iterations': iterations,
            'family_count': len(families), 'metrics': {key: {'difference': observed_b[key] - observed_a[key],
             'percentile_95_ci': [_percentile(sorted(values), .025), _percentile(sorted(values), .975)]}
             for key, values in differences.items()}}


def evaluate(gold, predictions):
    aligned = align(gold, predictions)
    result = summarize(aligned)
    result['subsets'] = {}
    for name, predicate in [('challenge', lambda r: r['family'].startswith('challenge_')),
                            ('controlled', lambda r: not r['family'].startswith('challenge_'))]:
        subset = [r for r in aligned if predicate(r)]
        if subset:
            result['subsets'][name] = summarize(subset)
    result['protocol'] = {
        'strict_output': 'Exactly two JSON keys decision and skill; known labels; null skill except supported replacement; no duplicate keys, fences, or prose.',
        'missing_predictions': 'Counted as invalid and incorrect; never dropped.',
        'duplicate_or_unknown_ids': 'Fatal input error.',
        'skill_denominator': 'All gold replace_task examples, including invalid predictions and wrong decisions.',
        'false_cancel_denominator': 'All gold non-cancel examples.',
        'missed_cancel_denominator': 'All gold cancel examples; invalid/missing predictions count as missed.',
        'prediction_only': 'Allowed for semantic metrics; unavailable raw output fails strict-format compliance (raw_output_coverage is reported).',
        'decision_macro_f1': 'Unweighted mean over all five fixed decisions; invalid outputs contribute false negatives.',
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare', help='Optional second JSONL; paired difference is comparison minus primary.')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    gold = load_gold(args.gold); predictions = load_predictions(args.predictions)
    result = evaluate(gold, predictions)
    if args.compare:
        comparison = load_predictions(args.compare)
        result['comparison'] = evaluate(gold, comparison)
        result['paired_bootstrap'] = paired_family_bootstrap(align(gold, predictions), align(gold, comparison), args.bootstrap, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('n', 'decision_accuracy', 'joint_accuracy', 'macro_f1', 'strict_json_validity', 'skill_accuracy')}, indent=2))

if __name__ == '__main__':
    main()
