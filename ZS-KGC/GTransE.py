import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Model import Model

class GTransE(Model):

	def __init__(self, ent_tot, rel_tot, pretrained, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(GTransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		if pretrained == None:
			self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		else:
			weights = torch.load(pretrained)
			weights = F.normalize(weights)
			self.ent_embeddings = nn.Embedding.from_pretrained(weights, freeze=False)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		X = self.ent_embeddings(torch.arange(self.ent_tot).cuda())
		h = X[batch_h]
		t = X[batch_t]
		r = self.rel_embeddings(batch_r)
		score = self._calc(h, t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()