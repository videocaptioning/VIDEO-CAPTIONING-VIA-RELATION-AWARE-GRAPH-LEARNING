import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import pickle
import time
from utils.utils import *
from utils.data import get_train_loader
from utils.opt import parse_opt
from reward import *
import models
import torch
import torch.nn as nn
import numpy as np
import sys
import random
from evaluate import evaluate, convert_data_to_coco_scorer_format

# self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

def save_checkpoint(model, optimizer, append=''):
    if len(append) > 0:
        append = '-' + append
    checkpoint_path = os.path.join(opt.result_dir, 'model%s.pth' % (append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.result_dir, 'optimizer%s.pth' % (append))
    torch.save(optimizer.state_dict(), optimizer_path)

def main(opt):
    """
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print("Vocabulary Size is: ", vocab_size)
    if not os.path.exists(opt.result_dir):
        os.mkdir(opt.result_dir)
    net = models.setup(opt, vocab)
    net.cuda()
    if opt.self_critical_training == 1:
        print('Loading Model From ', os.path.join(opt.start_from, 'model.pth'))
        net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)
        optimizer = ReduceLROnPlateau(optimizer, factor=0.625, patience=4)
        init_scorer(opt.cached_tokens)
        criterion = RewardCriterion()
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)

    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, opt.region_feature_h5_path, opt.rel_feature_h5_path, opt.tem_feature_h5_path, opt.train_gts, opt.rel_masks_h5_path, opt.train_batch_size)
    train_total_step = len(train_loader)

    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)
    saving_schedule = [int(x * train_total_step / opt.save_per_epoch) for x in list(range(1, opt.save_per_epoch + 1))]
    best_cider = 0.
    opt.sample_method = 'greedy'
    epsilon = 1.0
    for epoch in range(opt.max_epoch):
        total_loss = 0.
        if opt.learning_rate_decay and epoch % opt.learning_rate_decay_every == 0 and epoch > 0 and opt.self_critical_training == 0:
            opt.learning_rate /= opt.learning_rate_decay_rate
            set_lr(optimizer, opt.learning_rate)
        print('Epoch {} Learning Rate is {:.7f}'.format(epoch, get_lr(optimizer)))

        for iteration, train_data in enumerate(train_loader, start=1):
            start = time.time()
            sys.stdout.flush()
            frames, regions, rel_feats, tem_feats, spatials, captions, pos_tags, cap_lens, video_ids, gts, rel_masks = train_data
            frames = frames.cuda()
            regions = regions.cuda()
            rel_feats = rel_feats.cuda()
            tem_feats = tem_feats.cuda()
            captions = captions.cuda()
            rel_masks = rel_masks.cuda()

            optimizer.zero_grad()
            if opt.self_critical_training == 0:
                outputs = net(frames, regions, rel_feats, tem_feats, rel_masks, captions, epsilon)
                bsz = len(outputs)
                outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
                outputs = outputs.view(-1, vocab_size)
                targets = torch.cat([captions[j][:cap_lens[j]] for j in range(bsz)], 0)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                print('Epoch {} Iteration {}/{} Loss {:.3f} Time {:.3f}'.format(epoch, iteration, train_total_step, loss.item(), time.time() - start))
            else:
                beam_size = opt.beam_size
                opt.beam_size = 1
                net.eval()
                with torch.no_grad():
                    greedy_res, _ = net(frames, regions, rel_feats, tem_feats, rel_masks, None)
                net.train()
                opt.sample_method = 'sample'
                gen_result, gen_logprobs = net(frames, regions, rel_feats, tem_feats, rel_masks, None)
                reward = get_self_critical_reward(greedy_res, gts, gen_result, opt)
                reward = torch.from_numpy(reward).float().to(gen_result.device)
                loss = criterion(gen_logprobs.squeeze(-1), gen_result.data, reward)
                print('Epoch {} Iteration {}/{} Loss {:.3f} Reward {:.3f} Time {:.3f}'.format(epoch, iteration, train_total_step, loss.item(), reward[:, 0].mean(), time.time() - start))
                opt.sample_method = 'greedy'
                opt.beam_size = beam_size

            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            total_loss += loss.item()

            if iteration in saving_schedule:
                net.eval()
                metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference)
                print('Summary Epoch {} CIDEr {:.3f} BLEU_4 {:.3f}'.format(epoch, metrics['CIDEr'] * 100, metrics['Bleu_4'] * 100))
                net.train()
                save_checkpoint(net, optimizer)
                if opt.reward_metric == 'cider':
                    if metrics['CIDEr'] > best_cider:
                        best_cider = metrics['CIDEr']
                        print('Epoch {} Iteration {} Best CIDEr {:.3f}'.format(epoch, iteration, best_cider * 100))
                        save_checkpoint(net, optimizer, append='best')
                    if opt.self_critical_training:
                        optimizer.scheduler_step(-metrics['CIDEr'])
                elif opt.reward_metric == 'bleu':
                    if metrics['Bleu_4'] > best_cider:
                        best_cider = metrics['Bleu_4']
                        print('Epoch {} Iteration {} Best Bleu_4 {:.3f}'.format(epoch, iteration, best_cider * 100))
                        save_checkpoint(net, optimizer, append='best')
                    if opt.self_critical_training:
                        optimizer.scheduler_step(-metrics['Bleu_4'])
                elif opt.reward_metric == 'rouge':
                    if metrics['ROUGE_L'] > best_cider:
                        best_cider = metrics['ROUGE_L']
                        print('Epoch {} Iteration {} Best ROUGE_L {:.3f}'.format(epoch, iteration, best_cider * 100))
                        save_checkpoint(net, optimizer, append='best')
                    if opt.self_critical_training:
                        optimizer.scheduler_step(-metrics['ROUGE_L'])
                elif opt.reward_metric == 'meteor':
                    if metrics['METEOR'] > best_cider:
                        best_cider = metrics['METEOR']
                        print('Epoch {} Iteration {} Best METEOR {:.3f}'.format(epoch, iteration, best_cider * 100))
                        save_checkpoint(net, optimizer, append='best')
                    if opt.self_critical_training:
                        optimizer.scheduler_step(-metrics['METEOR'])


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
