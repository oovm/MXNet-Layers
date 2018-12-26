from mxnet.gluon import HybridBlock


class Pixel_Shuffle(HybridBlock):
	def __init__(self, up_factors, **kwargs):
		super(Pixel_Shuffle, self).__init__()
		self.up_factors = (2,2)

	def hybrid_forward(self, F,x):
		f1, f2 = self.up_factors
		x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C * f1 * f2, H, W) -> (N, C, f1 * f2, H, W)
		x = F.reshape(x, (0, 0, -3, 0))               # (N, C, f1 * f2, H, W)  -> (N, C, f1, f2, H, W)
		x = F.reshape(x, (0, 0, -4, -1, f2, 0))       # (N, C, f1, f2, H, W)   -> (N, C, H, f1, W, f2)
		x = F.reshape(x, (0, 0, 0, -3))               # (N, C, H, f1, W, f2)   -> (N, C, H * f1, W * f2)
		return x