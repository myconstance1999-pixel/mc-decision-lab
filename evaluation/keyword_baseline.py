"""Fixed, intentionally simple latest-utterance keyword baseline (no test fitting)."""
import argparse
import json
from pathlib import Path


def predict(utterance):
    # Precedence is fixed before measurement; this ignores dialogue and state.
    if any(word in utterance for word in ('改', '换', '跟随', '跟我走', '回家', '返回基地')):
        choices = [('follow_player', ('跟随', '跟我', '跟着我')),
                   ('return_home', ('回家', '返回', '回基地')),
                   ('store_items', ('存', '入箱', '入库')),
                   ('craft', ('制作', '合成', '做蛋糕', '做斧头')),
                   ('collect', ('收集', '采集', '去收', '去采'))]
        for skill, words in choices:
            if any(word in utterance for word in words):
                return {'decision': 'replace_task', 'skill': skill}
    if any(word in utterance for word in ('取消', '作废', '放弃', '撤销', '不做了', '不再', '彻底', '终止')):
        decision = 'cancel'
    elif any(word in utterance for word in ('暂停', '等我', '停一下', '停一会', '歇', '暂缓', '稍后', '等会')):
        decision = 'pause'
    elif any(word in utterance for word in ('那个', '另一', '哪', '没说', '未说', '你猜')):
        decision = 'clarify'
    else:
        decision = 'continue'
    return {'decision': decision, 'skill': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.gold).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            prediction = predict(json.loads(row['input'])['latest_utterance'])
            handle.write(json.dumps({'id': row['id'], 'raw_output': json.dumps(prediction, ensure_ascii=False, separators=(',', ':'))}, ensure_ascii=False) + '\n')
    print(f'Wrote {len(rows)} fixed-rule predictions to {output}')

if __name__ == '__main__':
    main()
