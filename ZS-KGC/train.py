import os
import argparse
import torch

import openke
from openke.config import Trainer, Tester
from openke.module.model import GTransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--pretrained')
	parser.add_argument('--dataset', default="ImageNet21K")
	parser.add_argument('--dim', type=int, default=2048)
	parser.add_argument('--gpu', default='0')

	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	dataset = args.dataset
	pretrained = args.pretrained
	dim = args.dim

	nbatches, neg_ent, margin, lr = 8, 5, 25.0, 0.1
	if dataset == "ImageNet21K":
		nbatches, neg_ent, margin, lr = 16, 10, 50.0, 0.5

	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = f"./{dataset}/", 
		nbatches = nbatches,
		threads = 4, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = neg_ent,
		neg_rel = 0)

	# dataloader for test
	test_dataloader = TestDataLoader(f"./{dataset}/", "link")

	# define the model
	transe = GTransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		pretrained = f"./{dataset}/{pretrained}",
		dim = dim, 
		p_norm = 2, 
		norm_flag = True,
	)

	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = margin),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = lr, use_gpu = True)
	trainer.run()
	transe.save_checkpoint('./checkpoint/transe.ckpt')

	# # test the model
	transe.load_checkpoint('./checkpoint/transe.ckpt')
	tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
	# tester.run_link_prediction(type_constrain=False)

	# ZSL Testing
	test_set = tester.data_loader
	zsl = set()
	for index, [data_head, data_tail] in enumerate(test_set):
		zsl = zsl.union(data_head['batch_t'])
	zsl = list(sorted(zsl))

	len_zsl = 150
	if dataset == "ImageNet21K":
		len_zsl = 1221
	len_test = len(test_set)
	mrr_z, mr_z, hit1_z, hit2_z, hit3_z, hit5_z, hit10_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

	for index, [data_head, data_tail] in enumerate(test_set):
		score = tester.test_one_step(data_tail)
		target = score[data_head['batch_t']]
		score_zsl = score[zsl]
		rk_zsl = sum(score_zsl <= target)

		mrr_z += 1/rk_zsl
		mr_z += rk_zsl
		if rk_zsl <= 1:
			hit1_z += 1
		if rk_zsl <= 2:
			hit2_z += 1	
		if rk_zsl <= 3:
			hit3_z += 1
		if rk_zsl <= 5:
			hit5_z += 1
		if rk_zsl <= 10:
			hit10_z += 1

	results = f"ZSL: \nMRR:{mrr_z/len_test:10.4f}  MR:{mr_z/len_test:10.4f}  Hit1:{hit1_z/len_test:10.4f}  Hit2:{hit2_z/len_test:10.4f}  Hit3:{hit3_z/len_test:10.4f}  Hit5:{hit5_z/len_test:10.4f}  Hit10:{hit10_z/len_test:10.4f}"
	print(results)


