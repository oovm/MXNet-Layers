from mxnet.gluon import HybridBlock


def channel_shuffle(x, groups):
	return x.reshape((0, -4, groups, -1, -2)).swapaxes(1, 2).reshape((0, -3, -2))


class Channel_Shuffle(HybridBlock):
	def __init__(self, channels, groups, **kwargs):
		super(Channel_Shuffle, self).__init__(**kwargs)
		assert (channels % groups == 0)
		self.groups = groups

	def hybrid_forward(self, F, x, **kwargs):
		return channel_shuffle(x, self.groups)


class Channel_Shuffle2(HybridBlock):
	def __init__(self, channels, groups, **kwargs):
		super(Channel_Shuffle2, self).__init__(**kwargs)
		assert (channels % groups == 0)
		self.channels_per_group = channels // groups

	def hybrid_forward(self, F, x, **kwargs):
		return channel_shuffle(x, self.channels_per_group)
