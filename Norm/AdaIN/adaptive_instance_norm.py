import mxnet as mx
from mxnet.gluon import HybridBlock


def mean_var_sigma(input, eps=1e-5):
	input_mean = input.mean(axis=(2, 3), keepdims=True)
	input_var = ((input - input_mean) ** 2).mean(axis=(2, 3), keepdims=True)
	input_sigma = mx.symbol.broadcast_plus(input_var, eps).sqrt()
	return input_mean, input_var, input_sigma


def AdaIN(content, style, eps=1e-5):
	content_mean, content_var, content_sigma = mean_var_sigma(content, eps)
	style_mean, style_var, style_sigma = mean_var_sigma(style, eps)
	return (content - content_mean) * style_sigma / content_sigma + style_mean


class Adaptive_Instance_Norm(HybridBlock):
	def __init__(self, content, style, **kwargs):
		super(Adaptive_Instance_Norm, self).__init__(**kwargs)
		self.eps = 1e-5

	def hybrid_forward(self, F, x, **kwargs):
		return AdaIN(content, style, self.eps)
