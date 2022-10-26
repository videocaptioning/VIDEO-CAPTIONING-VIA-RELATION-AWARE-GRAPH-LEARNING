import sys
sys.path.append("caption-eval")
import torch
import os
import pickle
import models
import json
from utils.utils import Vocabulary
from utils.data import get_eval_loader
from cocoeval import COCOScorer, suppress_stdout_stderr
from utils.opt import parse_opt


def convert_data_to_coco_scorer_format(reference):
    reference_json = {}
    non_ascii_count = 0
    with open(reference, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            try:
                sent.encode('ascii', 'ignore').decode('ascii')
            except UnicodeDecodeError:
                non_ascii_count += 1
                continue
            if vid in reference_json:
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
            else:
                reference_json[vid] = []
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print("=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20)
    return reference_json

def convert_prediction(prediction):
    prediction_json = {}
    with open(prediction, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            prediction_json[vid] = [{u'video_id': vid, u'caption': sent}]
    return prediction_json

def evaluate(opt, net, eval_range, prediction_txt_path, reference):
    eval_loader = get_eval_loader(eval_range, opt.feature_h5_path, opt.region_feature_h5_path, opt.rel_feature_h5_path, opt.tem_feature_h5_path, opt.rel_masks_h5_path, opt.test_batch_size)
    result = {}
    for iteration, val_data  in enumerate(eval_loader, start=1):
        frames, regions, rel_feats, tem_feats, spatials, video_ids, rel_masks = val_data
        frames = frames.cuda()
        regions = regions.cuda()
        rel_feats = rel_feats.cuda()
        tem_feats = tem_feats.cuda()
        rel_masks = rel_masks.cuda()

        outputs, _ = net(frames, regions, rel_feats, tem_feats, rel_masks, None)

        for (tokens, vid) in zip(outputs, video_ids):
            s = net.decoder.decode_tokens(tokens.data)
            result[vid] = s
            print('VID {}: {}'.format(vid, s))

    with open(prediction_txt_path, 'w') as f:
        for vid, s in result.items():
            f.write('%d\t%s\n' % (vid, s))

    prediction_json = convert_prediction(prediction_txt_path)
    scorer = COCOScorer()
    with suppress_stdout_stderr():
        scores, sub_category_score = scorer.score(reference, prediction_json, prediction_json.keys())
    for metric, score in scores.items():
        print('%s: %.6f' % (metric, score * 100))
    return scores

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# if __name__ == '__main__':
#     opt = parse_opt()
#     with open(opt.vocab_pkl_path, 'rb') as f:
#         vocab = pickle.load(f)
#     reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)
#     net = models.setup(opt, vocab).cuda()
#     opt.start_from = 'log/msrvtt/msrvtt_graph13_rouge/'
#     net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
#     net.eval()
#     print('Load Model from {}'.format(os.path.join(opt.start_from, 'model-best.pth')))
#     metrics = evaluate(opt, net, opt.test_range, 'test.txt', reference)
#     print(metrics['CIDEr'])


if __name__ == '__main__':
    prediction_txt_path = '/data0/zy/msvtt/features/msrvtt/msrvtt_test_references.txt'
    prediction_json_path = 'msrvtt_gt_json.json'


    def convert(prediction):
        prediction_json = {}
        i = 0
        with open(prediction, 'r') as f:
            lines = f.readlines()
            for line in lines:
                vid = line.split('\t')[0]
                sent = line.split('\t')[1].strip()
                prediction_json[i] = [{u'video_id': vid, u'caption': sent}]
                i = i + 1
        return prediction_json

    prediction_json = convert(prediction_txt_path)
    with open(prediction_json_path, 'w') as f:
        for k, v in prediction_json.items():
            json.dump(v, f)





